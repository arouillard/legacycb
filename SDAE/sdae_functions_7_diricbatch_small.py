# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import time
import shutil
import sdae_design_functions
import sdae_apply_functions
import datasetIO

 
def create_batch_ids(total_size, batch_size=100, dtype='int32'):
    # returns a numpy array of batch_ids
    batches = round(total_size/batch_size)
    batch_limits = np.linspace(0, total_size, batches+1, dtype=dtype)
    batch_ids = np.zeros(total_size, dtype=dtype)
    for i, (lb, ub) in enumerate(zip(batch_limits[:-1], batch_limits[1:])):
        batch_ids[lb:ub] = i
    return batch_ids

def create_variables(dimensions, initialization_distribution, initialization_sigma):
    # returns W, a list of weight matrices for encoding
    # (note this model will use W.T for decoding)
    # bencode, a list of bias vectors for encoding
    # bdecode, a list of bias vectors for decoding
    # all variables are randomly initialized according to initialization_distribution
    global_step = tf.Variable(0, trainable=False)
    W = []
    bencode = []
    bdecode = []
    for dim1, dim2 in zip(dimensions[:-1], dimensions[1:]):
        W.append(tf.Variable(initialization_distribution([dim1, dim2], stddev=initialization_sigma)))
        bencode.append(tf.Variable(initialization_distribution([1, dim2], stddev=initialization_sigma)))
        bdecode.append(tf.Variable(initialization_distribution([1, dim1], stddev=initialization_sigma)))
    return global_step, W, bencode, bdecode

def update_variables(dimensions, initialization_distribution, initialization_sigma, variables_path, fix_or_init, include_global_step):
    # updates W, bencode, and bdecode with values from a previous training run
    # previous values can be fixed for layerwise pretraining or initialized for finetuning
    with open(variables_path, 'rb') as fr:
        global_step_prev, W_prev, bencode_prev, bdecode_prev = pickle.load(fr)
    if include_global_step:
        global_step = tf.Variable(global_step_prev, trainable=False)
    else:
        global_step = tf.Variable(0, trainable=False)
    W = []
    bencode = []
    bdecode = []
    if fix_or_init == 'fix':
        for i, (w, be, bd) in enumerate(zip(W_prev, bencode_prev, bdecode_prev)):
            W.append(tf.Variable(w, trainable=False))
            bencode.append(tf.Variable(be, trainable=False))
            bdecode.append(tf.Variable(bd, trainable=False))
    elif fix_or_init == 'init':
        for i, (w, be, bd) in enumerate(zip(W_prev, bencode_prev, bdecode_prev)):
            W.append(tf.Variable(w))
            bencode.append(tf.Variable(be))
            bdecode.append(tf.Variable(bd))
    else:
        raise ValueError('fix_or_init must be fix or init')
    if len(W) < len(dimensions)-1:
        for dim1, dim2 in zip(dimensions[i+1:-1], dimensions[i+2:]):
            W.append(tf.Variable(initialization_distribution([dim1, dim2], stddev=initialization_sigma)))
            bencode.append(tf.Variable(initialization_distribution([1, dim2], stddev=initialization_sigma)))
            bdecode.append(tf.Variable(initialization_distribution([1, dim1], stddev=initialization_sigma)))
    return global_step, W, bencode, bdecode

def create_autoencoder(x, activation, apply_activation_to_output, apply_activation_to_embedding, use_batchnorm, is_training, W, bencode, bdecode):
    # returns h, a list of activations from input to bottleneck layer
    # hhat, a list of activations from bottleneck layer to output layer
    # xhat, a reference to the output layer (i.e. the reconstruction)
    if activation == tf.nn.relu:
        scale = False
    else:
        scale = True
    h = [x]
    for i, (w, be) in enumerate(zip(W, bencode)):
        if use_batchnorm:
            if i == len(W)-1 and not apply_activation_to_embedding:
                h.append(tf.layers.batch_normalization(tf.matmul(h[i], w), scale=False, training=is_training))
            else:
                h.append(activation(tf.layers.batch_normalization(tf.matmul(h[i], w), scale=scale, training=is_training)))
        else:
            if i == len(W)-1 and not apply_activation_to_embedding:
                h.append(tf.matmul(h[i], w) + be)
            else:
                h.append(activation(tf.matmul(h[i], w) + be))
    hhat = [h[-1]]
    for i, (w, bd) in enumerate(zip(W[::-1], bdecode[::-1])):
        if use_batchnorm:
            if i == len(W)-1 and not apply_activation_to_output:
                hhat.append(tf.layers.batch_normalization(tf.matmul(hhat[i], w, transpose_b=True), scale=False, training=is_training))
            else:
                hhat.append(activation(tf.layers.batch_normalization(tf.matmul(hhat[i], w, transpose_b=True), scale=scale, training=is_training)))
        else:
            if i == len(W)-1 and not apply_activation_to_output:
                hhat.append(tf.matmul(hhat[i], w, transpose_b=True) + bd)
            else:
                hhat.append(activation(tf.matmul(hhat[i], w, transpose_b=True) + bd))
    xhat = hhat[-1]
    return h, hhat, xhat

def apply_noise(x, noise, noise_mask, noise_operation):
    # returns the input corrupted by noise
    if noise_operation == 'add':
        return x + noise_mask*noise
    elif noise_operation == 'replace':
        return x + noise_mask*(noise - x)
    elif noise_operation == 'flip':
        return np.abs(x - noise_mask)
    else:
        raise ValueError('noise operation not specified')

def configure_session(processor, gpu_memory_fraction=0.2):
    # returns tensorflow configuration settings appropriate for CPU or GPU training
    if processor == 'cpu':
        session_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8) # this prevents hogging CPUs
    elif processor == 'gpu':
        session_config = tf.ConfigProto(allow_soft_placement=True) # can choose any available GPU
        session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction # this prevents hogging the GPU
    else:
        raise ValueError('invalid processor. specify cpu or gpu')
    return session_config




def main(d):
    # d is a dictionary containing the auto-encoder design specifications and training phase specifications

    # RESET DEFAULT GRAPH
    print('resetting default graph...', flush=True)    
    tf.reset_default_graph()
    
    
    
    
    # FINISH CONFIGURATION
    print('finishing configuration...', flush=True)
    
    # specify noise distribution
    if d['noise_distribution'] == 'truncnorm':
        noise_distribution = tf.truncated_normal
    elif d['noise_distribution'] == 'bernoulli':
        noise_distribution = tf.random_uniform
    elif d['noise_distribution'] == 'uniform':
        noise_distribution = tf.random_uniform
        
    # specify distribution of initial weights
    if d['initialization_distribution'] == 'truncnorm':
        initialization_distribution = tf.truncated_normal
    
    # specify activation function
    if d['activation_function'] == 'tanh':
        activation_function = {'tf':tf.tanh, 'np':sdae_apply_functions.tanh}
    elif d['activation_function'] == 'relu':
        activation_function = {'tf':tf.nn.relu, 'np':sdae_apply_functions.relu}
    elif d['activation_function'] == 'elu':
        activation_function = {'tf':tf.nn.elu, 'np':sdae_apply_functions.elu}
    elif d['activation_function'] == 'sigmoid':
        activation_function = {'tf':tf.sigmoid, 'np':sdae_apply_functions.sigmoid}

    # load data
    partitions = ['train', 'valid', 'test']
    dataset = {}
    for partition in partitions:
        if partition == 'train':
            dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], 'valid'))
            dataset[partition].append(datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], 'test')), 0)
        elif partition == 'valid':
            dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], 'train'))
        else:
            dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], partition))
        d['{0}_examples'.format(partition)] = dataset[partition].shape[0]
    
    # get loss weights
    posweights = 1/2/dataset['train'].matrix.sum(0, keepdims=True)
    posweights[0,dataset['train'].columnlabels == 'row_frac'] = 1/dataset['train'].matrix.shape[0]
    negweights = 1/2/(dataset['train'].matrix.shape[0] - dataset['train'].matrix.sum(0, keepdims=True))
    negweights[0,dataset['train'].columnlabels == 'row_frac'] = 1/dataset['train'].matrix.shape[0]
    marginalprobabilities = dataset['train'].matrix.mean(0, keepdims=True)
    
    
    # create output directory
    if not os.path.exists(d['output_path']):
        os.makedirs(d['output_path'])
            
    # initialize model architecture (number of layers and dimension of each layer)
    d['current_dimensions'] = d['all_dimensions'][:d['current_hidden_layer']+1] # dimensions of model up to current depth
    
    # specify embedding function for current training phase
    # we want the option of skipping the embedding activation function to apply only to the full model
    if not d['apply_activation_to_embedding'] and d['current_dimensions'] == d['all_dimensions']:
        d['current_apply_activation_to_embedding'] = False
    else:
        d['current_apply_activation_to_embedding'] = True
    
    # initialize assignments of training examples to mini-batches and number of training steps for stochastic gradient descent
    d['batch_size'] = d['batch_fraction']*d['train_examples']
    batch_ids = create_batch_ids(d['train_examples'], d['batch_size'])
    d['batches'] = np.unique(batch_ids).size
    d['steps'] = d['current_epochs']*d['batches']
    
    # specify path to weights from previous training run
    d['previous_variables_path'] = '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['previous_hidden_layer'], d['previous_finetuning_run'])
    d['fix_or_init'] = 'fix' if d['current_finetuning_run'] == 0 else 'init' # fix for pretraining, init for finetuning
    
    # specify rows and columns of figure showing data reconstructions
    d['reconstruction_rows'] = int(np.round(np.sqrt(np.min([100, d['valid_examples']])/2)))
    d['reconstruction_cols'] = 2*d['reconstruction_rows']
    
    # print some design information
    print('input path: {0}'.format(d['input_path']), flush=True)
    print('output path: {0}'.format(d['output_path']), flush=True)
    print('previous variables path: {0}'.format(d['previous_variables_path']), flush=True)
    print('previous variables fix or init: {0}'.format(d['fix_or_init']), flush=True)
    
    
    
    
    # SAVE CURRENT DESIGN
    print('saving current design...', flush=True)
    with open('{0}/design_layer{1!s}_finetuning{2!s}.json'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(d, fw, indent=2)
    
    
    
    
    # DEFINE REPORTING VARIABLES
    print('defining reporting variables...', flush=True)
    reporting_steps = sdae_design_functions.create_reporting_steps(d['steps'], d['firstcheckpoint'], d['maxstepspercheckpoint'])
    valid_losses = np.zeros(reporting_steps.size, dtype='float32')
    train_losses = np.zeros(reporting_steps.size, dtype='float32')
    valid_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
    train_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
    print('reporting steps:', reporting_steps, flush=True)
    
    
    
    
    # DEFINE COMPUTATIONAL GRAPH
    # define placeholders for input data, use None to allow feeding different numbers of examples
    print('defining placeholders...', flush=True)
    training = tf.placeholder(tf.bool, [])
    noise_stdv = tf.placeholder(tf.float32, [])
    noise_prob = tf.placeholder(tf.float32, [])
    dirichlet_log_sparsity_lb = tf.placeholder(tf.float32, [])
    dirichlet_log_sparsity_ub = tf.placeholder(tf.float32, [])
    dirichlet_batch_size = tf.placeholder(tf.int32, [])
    training_and_validation_data_initializer = tf.placeholder(tf.float32, [dataset['train'].shape[0]+dataset['valid'].shape[0], dataset['train'].shape[1]])
    selection_mask = tf.placeholder(tf.bool, [dataset['train'].shape[0]+dataset['valid'].shape[0]])
    pos_weights_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    neg_weights_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    marginal_probabilities_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])

    # define variables
    # W contains the weights, bencode contains the biases for encoding, and bdecode contains the biases for decoding
    print('defining variables...', flush=True)
    training_and_validation_data = tf.Variable(training_and_validation_data_initializer, trainable=False, collections=[])
    pos_weights = tf.Variable(pos_weights_initializer, trainable=False, collections=[])
    neg_weights = tf.Variable(neg_weights_initializer, trainable=False, collections=[])
    marginal_probabilities = tf.Variable(marginal_probabilities_initializer, trainable=False, collections=[])
    if os.path.exists(d['previous_variables_path']):
        # update variables (if continuing from a previous training run)
        print('loading previous variables...', flush=True)
        global_step, W, bencode, bdecode = update_variables(d['current_dimensions'], initialization_distribution, d['initialization_sigma'], d['previous_variables_path'], d['fix_or_init'], d['include_global_step'])
    elif (d['current_hidden_layer'] == 1 and d['current_finetuning_run'] == 0) or d['skip_layerwise_training']:
        # create variables        
        global_step, W, bencode, bdecode = create_variables(d['current_dimensions'], initialization_distribution, d['initialization_sigma'])
    else:
        raise ValueError('could not find previous variables')

    # define model
    # h contains the activations from input layer to bottleneck layer
    # hhat contains the activations from bottleneck layer to output layer
    # xhat is a reference to the output layer (i.e. the reconstruction)
    print('defining model...', flush=True)
    def ftrue():
        training_data = tf.boolean_mask(training_and_validation_data, selection_mask)
        dirichlet_sparsities = 10**(tf.random_uniform((dirichlet_batch_size, 1), minval=dirichlet_log_sparsity_lb, maxval=dirichlet_log_sparsity_ub))
        dirichlet_concentrations = dirichlet_sparsities*tf.ones((dirichlet_batch_size, tf.shape(training_data)[0]), tf.float32)
        dirichlet_distributions = tf.distributions.Dirichlet(dirichlet_concentrations)
        dirichlet_samples = dirichlet_distributions.sample()
        return tf.matmul(dirichlet_samples, training_data)
    def ffalse():
        return tf.boolean_mask(training_and_validation_data, selection_mask)
    x = tf.cond(training, ftrue, ffalse)
#    if training:
#        training_data = tf.boolean_mask(training_and_validation_data, selection_mask)
#        dirichlet_sparsities = 10**(tf.random_uniform((dirichlet_batch_size, 1), minval=dirichlet_log_sparsity_lb, maxval=dirichlet_log_sparsity_ub))
#        dirichlet_concentrations = dirichlet_sparsities*tf.ones((dirichlet_batch_size, tf.shape(training_data)[0]), tf.float32)
#        dirichlet_distributions = tf.distributions.Dirichlet(dirichlet_concentrations)
#        dirichlet_samples = dirichlet_distributions.sample()
#        x = tf.matmul(dirichlet_samples, training_data)
#    else:
#        x = tf.boolean_mask(training_and_validation_data, selection_mask)
    loss_weights = pos_weights*x + neg_weights*(1-x)
    if d['noise_distribution'] == 'truncnorm':
        noise = noise_distribution(tf.shape(x), stddev=noise_stdv)
    elif d['noise_distribution'] == 'bernoulli':
        noise = tf.to_float(noise_distribution(tf.shape(x), minval=0, maxval=1) <= marginal_probabilities) # marginal
#        noise = tf.to_float(noise_distribution(tf.shape(x), minval=0, maxval=1) > marginal_probabilities) # 1 - marginal
    else:
        noise = noise_distribution(tf.shape(x), minval=0, maxval=noise_stdv)
    noise_mask = tf.to_float(tf.random_uniform(tf.shape(x)) <= noise_prob)
    xnoisy = apply_noise(x, noise, noise_mask, d['noise_operation'])
    if d['activation_function'] == 'sigmoid' and d['apply_activation_to_output']:
        h, hhat, xhat = create_autoencoder(xnoisy, activation_function['tf'], False, d['current_apply_activation_to_embedding'], d['use_batchnorm'], training, W, bencode, bdecode)
    else:
        h, hhat, xhat = create_autoencoder(xnoisy, activation_function['tf'], d['apply_activation_to_output'], d['current_apply_activation_to_embedding'], d['use_batchnorm'], training, W, bencode, bdecode)
    
    # define loss
    print('defining loss...', flush=True)
    if d['activation_function'] == 'sigmoid' and d['apply_activation_to_output']:
        loss = tf.reduce_mean(loss_weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=xhat))
    else:
        loss = tf.reduce_mean(tf.squared_difference(x, xhat)) # squared error loss

    # define optimizer and training function
    print('defining optimizer and training function...', flush=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=d['learning_rate'], epsilon=d['epsilon'], beta1=d['beta1'], beta2=d['beta2'])
    train_ops = optimizer.minimize(loss, global_step=global_step)
    
    # define update ops and add to train ops (if using batch norm)
    if d['use_batchnorm']:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_ops = [train_ops, update_ops]
    
    # collect batch norm variables
    if d['use_batchnorm']:
        bn_gammas = tf.global_variables(scope='batch_normalization.{0,2}/gamma:0')
        print(bn_gammas, flush=True)
        bn_betas = tf.global_variables(scope='batch_normalization.{0,2}/beta:0')
        bn_moving_means = tf.global_variables(scope='batch_normalization.{0,2}/moving_mean:0')
        bn_moving_variances = tf.global_variables(scope='batch_normalization.{0,2}/moving_variance:0')
    
    # define bottleneck layer preactivation
#    bottleneck_preactivation = tf.matmul(h[-2], W[-1]) + bencode[-1]


        
        
    # INITIALIZE TENSORFLOW SESSION
    print('initializing tensorflow session...', flush=True)
    init = tf.global_variables_initializer()
    session_config = configure_session(d['processor'], d['gpu_memory_fraction'])
    with tf.Session(config=session_config) as sess:
        sess.run(init)
       
        

        
        # TRAINING
        print('training...', flush=True)
        sess.run(training_and_validation_data.initializer, feed_dict={training_and_validation_data_initializer: np.append(dataset['train'].matrix, dataset['valid'].matrix, 0)})
        sess.run(pos_weights.initializer, feed_dict={pos_weights_initializer: posweights})
        sess.run(neg_weights.initializer, feed_dict={neg_weights_initializer: negweights})
        sess.run(marginal_probabilities.initializer, feed_dict={marginal_probabilities_initializer: marginalprobabilities})
        validation_id = -1
        batch_and_validation_ids = np.full(dataset['train'].shape[0]+dataset['valid'].shape[0], validation_id, dtype=batch_ids.dtype)
        is_train = np.append(np.ones(dataset['train'].shape[0], dtype='bool'), np.zeros(dataset['valid'].shape[0], dtype='bool'))
        is_valid = ~is_train
        training_step = 0
        i = 0
        overfitting_score = 0
        stopearly = False
        starttime = time.time()
        
        with open('{0}/log_layer{1!s}_finetuning{2!s}.txt'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), mode='wt', buffering=1) as fl:
            fl.write('\t'.join(['step', 'train_loss', 'valid_loss', 'train_noisy_loss', 'valid_noisy_loss', 'time']) + '\n')
            
            for epoch in range(d['current_epochs']):
                if stopearly:
                    break
                    
                # randomize assignment of training examples to batches
                np.random.shuffle(batch_ids)
                batch_and_validation_ids[is_train] = batch_ids
                
                for batch in range(d['batches']):
                    training_step += 1
                    
                    # select mini-batch
                    selected = batch_and_validation_ids == batch
                    
                    # update weights
                    sess.run(train_ops, feed_dict={training:True, selection_mask:selected, noise_prob:d['noise_probability'], noise_stdv:d['noise_sigma'], dirichlet_log_sparsity_lb:-4.0, dirichlet_log_sparsity_ub:0.0, dirichlet_batch_size:d['batch_size']})
                    
                    # record training and validation errors
                    if training_step == reporting_steps[i]:
                        train_losses[i] = sess.run(loss, feed_dict={training:False, selection_mask:is_train, noise_prob:0, noise_stdv:0, dirichlet_log_sparsity_lb:-4.0, dirichlet_log_sparsity_ub:0.0, dirichlet_batch_size:d['batch_size']})
                        train_noisy_losses[i] = sess.run(loss, feed_dict={training:False, selection_mask:is_train, noise_prob:d['noise_probability'], noise_stdv:d['noise_sigma'], dirichlet_log_sparsity_lb:-4.0, dirichlet_log_sparsity_ub:0.0, dirichlet_batch_size:d['batch_size']})
                        valid_losses[i] = sess.run(loss, feed_dict={training:False, selection_mask:is_valid, noise_prob:0, noise_stdv:0, dirichlet_log_sparsity_lb:-4.0, dirichlet_log_sparsity_ub:0.0, dirichlet_batch_size:d['batch_size']})
                        valid_noisy_losses[i] = sess.run(loss, feed_dict={training:False, selection_mask:is_valid, noise_prob:d['noise_probability'], noise_stdv:d['noise_sigma'], dirichlet_log_sparsity_lb:-4.0, dirichlet_log_sparsity_ub:0.0, dirichlet_batch_size:d['batch_size']})
                        print('step:{0:1.6g}, train loss:{1:1.3g}, valid loss:{2:1.3g}, train noisy loss:{3:1.3g},valid noisy loss:{4:1.3g}, time:{5:1.6g}'.format(reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], time.time() - starttime), flush=True)
                        fl.write('\t'.join(['{0:1.6g}'.format(x) for x in [reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], time.time() - starttime]]) + '\n')
                            
                        # save current weights, reconstructions, and projections
                        if training_step >= d['startsavingstep'] or training_step == reporting_steps[-1]:
                            with open('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], training_step), 'wb') as fw:
                                pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)
                            if d['use_batchnorm']:
                                with open('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], training_step), 'wb') as fw:
                                    pickle.dump((sess.run(bn_gammas), sess.run(bn_betas), sess.run(bn_moving_means), sess.run(bn_moving_variances)), fw)

                            # stop early if overfitting
                            if valid_losses[i] >= 1.01*(np.insert(valid_losses[:i], 0, np.inf).min()):
                                overfitting_score += 1
                            else:
                                overfitting_score = 0
                            if overfitting_score == d['overfitting_score_max']:
                                stopearly = True
                                print('stopping early!', flush=True)
                                break
                        i += 1
                        
        # end tensorflow session
        print('closing tensorflow session...', flush=True)
                        
                        
                        
                        
    # ROLL BACK IF OVERFITTING
    if stopearly:
        print('rolling back...', flush=True)
        reporting_steps = reporting_steps[:i+1]
        train_losses = train_losses[:i+1]
        valid_losses = valid_losses[:i+1]
        train_noisy_losses = train_noisy_losses[:i+1]
        valid_noisy_losses = valid_noisy_losses[:i+1]
#        selected_step = max([reporting_steps[i-d['overfitting_score_max']], d['startsavingstep']])
    else:
        print('completed all training steps...', flush=True)
#        selected_step = reporting_steps[-1]
    selected_step = min([max([reporting_steps[np.argmin(valid_losses)], d['startsavingstep']]), reporting_steps[-1]])
    print('selected step:{0}...'.format(selected_step), flush=True)
    
    
    
    
    # SAVE RESULTS
    print('saving results...', flush=True)
    with open('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'wb') as fw:
        pickle.dump({'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses, 'valid_noisy_losses':valid_noisy_losses, 'train_noisy_losses':train_noisy_losses}, fw)
    if d['current_dimensions'] == d['all_dimensions'] and (not d['use_finetuning'] or d['current_finetuning_run'] > 0):
        shutil.copyfile('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                        '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
        if d['use_batchnorm']:
            shutil.copyfile('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                            '{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
    else:
        shutil.move('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                    '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
        if d['use_batchnorm']:
            shutil.move('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                        '{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
    with open('{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
        W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
    if d['use_batchnorm']:
        with open('{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
            batchnorm_variables = pickle.load(fr) # gammas, betas, moving_means, moving_variances
        batchnorm_encode_variables, batchnorm_decode_variables = sdae_apply_functions.align_batchnorm_variables(batchnorm_variables, d['current_apply_activation_to_embedding'], d['apply_activation_to_output'])
    recon = {}
    embed = {}
    error = {}
    embed_preactivation = {}
    for partition in partitions:
        if d['use_batchnorm']:
            recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
            embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
        else:
            recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True)
            embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
        print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)
        if d['current_dimensions'] == d['all_dimensions'] and (not d['use_finetuning'] or d['current_finetuning_run'] > 0):
            datasetIO.save_datamatrix('{0}/{1}_embedding_layer{2!s}_finetuning{3!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed[partition])
            datasetIO.save_datamatrix('{0}/{1}_embedding_layer{2!s}_finetuning{3!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed[partition])
            if d['current_apply_activation_to_embedding']:
                datasetIO.save_datamatrix('{0}/{1}_embedding_preactivation_layer{2!s}_finetuning{3!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed_preactivation[partition])
                datasetIO.save_datamatrix('{0}/{1}_embedding_preactivation_layer{2!s}_finetuning{3!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed_preactivation[partition])
            
    
    
        
    # PLOT LOSS
    print('plotting loss...', flush=True)
    fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
    ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
    ax.semilogx(reporting_steps, train_losses, ':r', linewidth=1, label='train')
    ax.semilogx(reporting_steps, valid_losses, '-g', linewidth=1, label='valid')
    ax.semilogx(reporting_steps, train_noisy_losses, '--b', linewidth=1, label='train,noisy')
    ax.semilogx(reporting_steps, valid_noisy_losses, '-.k', linewidth=1, label='valid,noisy')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylabel('loss', fontsize=8)
    ax.set_xlabel('steps (selected step:{0!s})'.format(selected_step), fontsize=8)
    ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
    # ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', left=True, right=True, bottom=True, top=False,
                   labelleft=True, labelright=False, labelbottom=True, labeltop=False, labelsize=8)
    fg.savefig('{0}/optimization_path_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
    plt.close()
        
        
        
        
    # PLOT RECONSTRUCTIONS
    print('plotting reconstructions...', flush=True)
    num_recons = min([d['reconstruction_rows']*d['reconstruction_cols'], dataset['valid'].shape[0]])
    x_valid = dataset['valid'].matrix[:num_recons,:]
    xr_valid = recon['valid'].matrix[:num_recons,:]
    if x_valid.shape[1] > 1000:
        x_valid = x_valid[:,:1000]
        xr_valid = xr_valid[:,:1000]
    lb = np.append(x_valid, xr_valid, 1).min(1)
    ub = np.append(x_valid, xr_valid, 1).max(1)
    if d['apply_activation_to_output']:
        if d['activation_function'] == 'sigmoid':
            lb[:] = -0.05
            ub[:] = 1.05
        elif d['activation_function'] == 'tanh':
            lb[:] = -1.05
            ub[:] = 1.05
    fg, axs = plt.subplots(d['reconstruction_rows'], d['reconstruction_cols'], figsize=(6.5,3.25))
    for i, ax in enumerate(axs.reshape(-1)):
        if i < num_recons:
            ax.plot(x_valid[i,:], xr_valid[i,:], 'ok', markersize=0.5, markeredgewidth=0, alpha=0.1)
            ax.set_ylim(lb[i], ub[i])
            ax.set_xlim(lb[i], ub[i])
            ax.tick_params(axis='both', which='major', left=False, right=False, bottom=False, top=False, labelleft=False, labelright=False, labelbottom=False, labeltop=False, pad=4)
            ax.set_frame_on(False)
            ax.axvline(lb[i], linewidth=1, color='k')
            ax.axvline(ub[i], linewidth=1, color='k')
            ax.axhline(lb[i], linewidth=1, color='k')
            ax.axhline(ub[i], linewidth=1, color='k')
        else:
            fg.delaxes(ax)
    fg.savefig('{0}/reconstructions_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=1200)
    plt.close()
    
    
    
    
    # PLOT 2D EMBEDDING
    if d['current_dimensions'][-1] == 2  and (not d['use_finetuning'] or d['current_finetuning_run'] > 0):
        print('plotting 2d embedding...', flush=True)
        fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
        ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
        ax.plot(embed['train'].matrix[:,0], embed['train'].matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
        ax.plot(embed['valid'].matrix[:,0], embed['valid'].matrix[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
        ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False, pad=4)
        ax.set_frame_on(False)
        fg.savefig('{0}/embedding_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
        plt.close()
        
        if d['current_apply_activation_to_embedding']:
            fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
            ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
            ax.plot(embed_preactivation['train'].matrix[:,0], embed_preactivation['train'].matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.plot(embed_preactivation['valid'].matrix[:,0], embed_preactivation['valid'].matrix[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
            ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False, pad=4)
            ax.set_frame_on(False)
            fg.savefig('{0}/embedding_preactivation_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
            plt.close()
        
        
        
        
    print('done training phase.', flush=True)

    return d['current_hidden_layer'], d['current_finetuning_run'], d['current_epochs']
