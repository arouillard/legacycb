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
import copy
import pickle
import json
import os
import time
import shutil
import tsdae_design_functions
import tsdae_apply_functions
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
        
    # specify distribution of initial weights
    if d['initialization_distribution'] == 'truncnorm':
        initialization_distribution = tf.truncated_normal
    
    # specify activation function
    if d['activation_function'] == 'tanh':
        activation_function = {'tf':tf.tanh, 'np':tsdae_apply_functions.tanh}
    elif d['activation_function'] == 'relu':
        activation_function = {'tf':tf.nn.relu, 'np':tsdae_apply_functions.relu}
    elif d['activation_function'] == 'elu':
        activation_function = {'tf':tf.nn.elu, 'np':tsdae_apply_functions.elu}
    elif d['activation_function'] == 'sigmoid':
        activation_function = {'tf':tf.sigmoid, 'np':tsdae_apply_functions.sigmoid}

    # load data
    partitions = ['train', 'valid', 'test']
    dataset = {}
    for partition in partitions:
        dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], partition))
        d['{0}_examples'.format(partition)] = dataset[partition].shape[0]
        if 'all' not in dataset:
            dataset['all'] = copy.deepcopy(dataset[partition])
        else:
            dataset['all'].append(dataset[partition], 0)
    
    # get loss weights
    # we have features with mixed variable types and mixed missingness
    # strategy is to apply weights do the data points such that each feature has total weight of 1
    # for binary features (columnmeta['likelihood'] == 'bernoulli'), balance the weight on the positive and negative classes
    # for other features, uniform weight
    zero = 0.
    half = 0.5
    one = 1.
    true = True
    posweights = 1/2/(1 + np.nansum(dataset['train'].matrix, 0, keepdims=True))
    posweights[:,dataset['train'].columnmeta['likelihood'] != 'bernoulli'] = 1/np.sum(~np.isnan(dataset['train'].matrix[:,dataset['train'].columnmeta['likelihood'] != 'bernoulli']), 0, keepdims=True)
    negweights = 1/2/(1 + np.sum(~np.isnan(dataset['train'].matrix), 0, keepdims=True) - np.nansum(dataset['train'].matrix, 0, keepdims=True))
    negweights[:,dataset['train'].columnmeta['likelihood'] != 'bernoulli'] = 1/np.sum(~np.isnan(dataset['train'].matrix[:,dataset['train'].columnmeta['likelihood'] != 'bernoulli']), 0, keepdims=True)
    print('posweights nan:', np.isnan(posweights).any(), flush=True)
    print('negweights nan:', np.isnan(negweights).any(), flush=True)
    u_dataset, c_dataset = np.unique(dataset['train'].columnmeta['dataset'], return_counts=True)
    datasetweights = np.zeros((1, dataset['train'].shape[1]), dtype='float64')
    for dataset_name, dataset_count in zip(u_dataset, c_dataset):
        datasetweights[:,dataset['train'].columnmeta['dataset'] == dataset_name] = 1/u_dataset.size/dataset_count
        
    
    # get parameters for marginal distributions
    # will sample from marginal distributions to impute missing values
    # as well as to replace known values with corrupted values
    # for binary features, model as bernoulli (columnmeta['likelihood'] == 'bernoulli')
    # for other features, model as gaussian
    marginalprobabilities = (1 + np.nansum(dataset['train'].matrix, 0, keepdims=True))/(2 + np.sum(~np.isnan(dataset['train'].matrix), 0, keepdims=True)) # posterior mean of beta-bernoulli with prior a=b=1
    marginalstdvs = np.nanstd(dataset['train'].matrix, 0, keepdims=True)
    isbernoullimarginal = (dataset['train'].columnmeta['likelihood'] == 'bernoulli').astype('float64').reshape(1,-1)
    print('marginalprobabilities nan:', np.isnan(marginalprobabilities).any(), flush=True)
    print('marginalstdvs nan:', np.isnan(marginalstdvs).any(), flush=True)
    print('isbernoullimarginal nan:', np.isnan(isbernoullimarginal).any(), flush=True)
    
    # assign friendly nan value
    nanvalue = -666.666
    for partition in partitions:
        dataset[partition].matrix[np.isnan(dataset[partition].matrix)] = nanvalue
        
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
    reporting_steps = tsdae_design_functions.create_reporting_steps(d['steps'], d['firstcheckpoint'], d['maxstepspercheckpoint'])
    valid_losses = np.zeros(reporting_steps.size, dtype='float32')
    train_losses = np.zeros(reporting_steps.size, dtype='float32')
    valid_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
    train_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
    valid_losses_normal = np.zeros(reporting_steps.size, dtype='float32')
    train_losses_normal = np.zeros(reporting_steps.size, dtype='float32')
    valid_noisy_losses_normal = np.zeros(reporting_steps.size, dtype='float32')
    train_noisy_losses_normal = np.zeros(reporting_steps.size, dtype='float32')
    valid_losses_bernoulli = np.zeros(reporting_steps.size, dtype='float32')
    train_losses_bernoulli = np.zeros(reporting_steps.size, dtype='float32')
    valid_noisy_losses_bernoulli = np.zeros(reporting_steps.size, dtype='float32')
    train_noisy_losses_bernoulli = np.zeros(reporting_steps.size, dtype='float32')
    print('reporting steps:', reporting_steps, flush=True)
    
    
    
    
    # DEFINE COMPUTATIONAL GRAPH
    # define placeholders for input data, use None to allow feeding different numbers of examples
    print('defining placeholders...', flush=True)
    training = tf.placeholder(tf.bool, [])
    get_normal_to_bernoulli_error = tf.placeholder(tf.bool, [])
    noise_prob = tf.placeholder(tf.float32, [])
    structured_noise_prob = tf.placeholder(tf.float32, [])
    training_and_validation_data_initializer = tf.placeholder(tf.float32, [dataset['train'].shape[0]+dataset['valid'].shape[0], dataset['train'].shape[1]])
    selection_mask = tf.placeholder(tf.bool, [dataset['train'].shape[0]+dataset['valid'].shape[0]])
    pos_weights_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    neg_weights_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    dataset_weights_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    marginal_probabilities_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    marginal_stdvs_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    is_bernoulli_marginal_initializer = tf.placeholder(tf.float32, [1, dataset['train'].shape[1]])
    zero_initializer = tf.placeholder(tf.float32, [])
    half_initializer = tf.placeholder(tf.float32, [])
    one_initializer = tf.placeholder(tf.float32, [])
    nan_value_initializer = tf.placeholder(tf.float32, [])
    bernoulli_weight_initializer = tf.placeholder(tf.float32, [])
    
    # define variables
    # W contains the weights, bencode contains the biases for encoding, and bdecode contains the biases for decoding
    print('defining variables...', flush=True)
    training_and_validation_data = tf.Variable(training_and_validation_data_initializer, trainable=False, collections=[])
    pos_weights = tf.Variable(pos_weights_initializer, trainable=False, collections=[])
    neg_weights = tf.Variable(neg_weights_initializer, trainable=False, collections=[])
    dataset_weights = tf.Variable(dataset_weights_initializer, trainable=False, collections=[])
    marginal_probabilities = tf.Variable(marginal_probabilities_initializer, trainable=False, collections=[])
    marginal_stdvs = tf.Variable(marginal_stdvs_initializer, trainable=False, collections=[])
    is_bernoulli_marginal = tf.Variable(is_bernoulli_marginal_initializer, trainable=False, collections=[])
    zero_ = tf.Variable(zero_initializer, trainable=False, collections=[])
    half_ = tf.Variable(half_initializer, trainable=False, collections=[])
    one_ = tf.Variable(one_initializer, trainable=False, collections=[])
    nan_value = tf.Variable(nan_value_initializer, trainable=False, collections=[])
    bernoulli_weight = tf.Variable(bernoulli_weight_initializer, trainable=False, collections=[])
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
    x = tf.boolean_mask(training_and_validation_data, selection_mask)
    is_positive = tf.to_float(tf.greater(x, zero_))
    is_missing = tf.to_float(tf.equal(x, nan_value))
    is_complete_row = tf.to_float(tf.reduce_all(tf.not_equal(x, nan_value), axis=tf.to_int32(one_), keepdims=True))
    loss_weights = (pos_weights*is_positive + neg_weights*(one_-is_positive))*(one_-is_missing)*dataset_weights # missing values won't be included in loss calculation
    loss_weights = loss_weights/tf.reduce_mean(loss_weights)
    normal_loss_weights = loss_weights*(one_-is_bernoulli_marginal)
#    bernoulli_loss_weights = loss_weights*is_bernoulli_marginal
    bernoulli_loss_weights = tf.cond(get_normal_to_bernoulli_error, lambda: is_complete_row*loss_weights*is_bernoulli_marginal, lambda: loss_weights*is_bernoulli_marginal)
    normal_noise = tf.truncated_normal(tf.shape(x), mean=zero_, stddev=one_)*marginal_stdvs
    bernoulli_noise = tf.to_float(tf.random_uniform(tf.shape(x), minval=zero_, maxval=one_) <= marginal_probabilities)
    noise = bernoulli_noise*is_bernoulli_marginal + normal_noise*(one_-is_bernoulli_marginal)
    random_noise_mask = tf.to_float(tf.random_uniform(tf.shape(x)) <= noise_prob) # replace missing values and random fraction of known values with noise
#    structured_noise_mask = is_complete_row*tf.to_float(tf.random_uniform((tf.shape(x)[tf.to_int32(zero_)], tf.to_int32(one_))) <= structured_noise_prob)*tf.abs(tf.to_float(tf.random_uniform((tf.shape(x)[tf.to_int32(zero_)], tf.to_int32(one_))) <= half_) - is_bernoulli_marginal)
    structured_noise_mask = tf.cond(get_normal_to_bernoulli_error, lambda: is_complete_row*is_bernoulli_marginal, lambda: is_complete_row*tf.to_float(tf.random_uniform((tf.shape(x)[tf.to_int32(zero_)], tf.to_int32(one_))) <= structured_noise_prob)*tf.abs(tf.to_float(tf.random_uniform((tf.shape(x)[tf.to_int32(zero_)], tf.to_int32(one_))) > bernoulli_weight) - is_bernoulli_marginal))    
    noise_mask = random_noise_mask + structured_noise_mask - (random_noise_mask*structured_noise_mask)
    x = x + is_missing*(noise - x)
    xnoisy = x + noise_mask*(noise - x)
    h, hhat, xhat_preactivation = create_autoencoder(xnoisy, activation_function['tf'], False, d['current_apply_activation_to_embedding'], d['use_batchnorm'], training, W, bencode, bdecode)
    normal_loss = tf.reduce_sum(normal_loss_weights*tf.squared_difference(x, xhat_preactivation))/tf.reduce_sum(normal_loss_weights)
    bernoulli_loss = tf.reduce_sum(bernoulli_loss_weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=xhat_preactivation))/tf.reduce_sum(bernoulli_loss_weights)
#    loss = normal_loss + bernoulli_loss
#    loss = (one_-bernoulli_weight)*normal_loss + bernoulli_weight*bernoulli_loss
    loss = tf.cond(get_normal_to_bernoulli_error, lambda: bernoulli_loss, lambda: (one_-bernoulli_weight)*normal_loss + bernoulli_weight*bernoulli_loss)
    # make validation compute only normal to bernoulli error --> v12
    # yes, want validation to compute normal to bernoulli error, then experiment with training - how much to focus on bernoulli error and how much structured noise?
    # in the extreme of total structured noise and total focus on bernoulli error, then we are basically just looking at a standard ff neural net transforming normal to bernoulli
    # the question to answer is, does it help to have some reconstruction objective and some bernoulli to normal objective
    
    # remember for the current results to look look at performance with truely random imputation (not using marginal distributions)
    
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
        sess.run(dataset_weights.initializer, feed_dict={dataset_weights_initializer: datasetweights})
        sess.run(marginal_probabilities.initializer, feed_dict={marginal_probabilities_initializer: marginalprobabilities})
        sess.run(marginal_stdvs.initializer, feed_dict={marginal_stdvs_initializer: marginalstdvs})
        sess.run(is_bernoulli_marginal.initializer, feed_dict={is_bernoulli_marginal_initializer: isbernoullimarginal})
        sess.run(zero_.initializer, feed_dict={zero_initializer: zero})
        sess.run(half_.initializer, feed_dict={half_initializer: half})
        sess.run(one_.initializer, feed_dict={one_initializer: one})
        sess.run(nan_value.initializer, feed_dict={nan_value_initializer: nanvalue})
        sess.run(bernoulli_weight.initializer, feed_dict={bernoulli_weight_initializer: d['bernoulli_weight']})
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
            fl.write('\t'.join(['step', 'train_loss', 'valid_loss', 'train_noisy_loss', 'valid_noisy_loss', 'train_loss_normal', 'valid_loss_normal', 'train_noisy_loss_normal', 'valid_noisy_loss_normal', 'train_loss_bernoulli', 'valid_loss_bernoulli', 'train_noisy_loss_bernoulli', 'valid_noisy_loss_bernoulli', 'time']) + '\n')
            
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
                    sess.run(train_ops, feed_dict={training:True, get_normal_to_bernoulli_error:False, selection_mask:selected, noise_prob:d['noise_probability'], structured_noise_prob:d['structured_noise_probability']})
                    
                    # record training and validation errors
                    if training_step == reporting_steps[i]:
                        train_losses[i], train_losses_normal[i], train_losses_bernoulli[i] = sess.run([loss, normal_loss, bernoulli_loss], feed_dict={training:False, get_normal_to_bernoulli_error:False, selection_mask:is_train, noise_prob:0, structured_noise_prob:0})
                        train_noisy_losses[i], train_noisy_losses_normal[i], train_noisy_losses_bernoulli[i] = sess.run([loss, normal_loss, bernoulli_loss], feed_dict={training:False, get_normal_to_bernoulli_error:False, selection_mask:is_train, noise_prob:d['noise_probability'], structured_noise_prob:d['structured_noise_probability']})
                        valid_losses[i], valid_losses_normal[i], valid_losses_bernoulli[i] = sess.run([loss, normal_loss, bernoulli_loss], feed_dict={training:False, get_normal_to_bernoulli_error:True, selection_mask:is_valid, noise_prob:0, structured_noise_prob:0})
                        valid_noisy_losses[i], valid_noisy_losses_normal[i], valid_noisy_losses_bernoulli[i] = sess.run([loss, normal_loss, bernoulli_loss], feed_dict={training:False, get_normal_to_bernoulli_error:True, selection_mask:is_valid, noise_prob:d['noise_probability'], structured_noise_prob:d['structured_noise_probability']})
                        print('step:{0:1.6g}, trn:{1:1.3g}, vld:{2:1.3g}, trnn:{3:1.3g}, vldn:{4:1.3g}, trnN:{5:1.3g}, vldN:{6:1.3g}, trnnN:{7:1.3g}, vldnN:{8:1.3g}, trnB:{9:1.3g}, vldB:{10:1.3g}, trnnB:{11:1.3g}, vldnB:{12:1.3g}, time:{13:1.6g}'.format(reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], train_losses_normal[i], valid_losses_normal[i], train_noisy_losses_normal[i], valid_noisy_losses_normal[i], train_losses_bernoulli[i], valid_losses_bernoulli[i], train_noisy_losses_bernoulli[i], valid_noisy_losses_bernoulli[i], time.time() - starttime), flush=True)
                        fl.write('\t'.join(['{0:1.6g}'.format(x) for x in [reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], train_losses_normal[i], valid_losses_normal[i], train_noisy_losses_normal[i], valid_noisy_losses_normal[i], train_losses_bernoulli[i], valid_losses_bernoulli[i], train_noisy_losses_bernoulli[i], valid_noisy_losses_bernoulli[i], time.time() - starttime]]) + '\n')
                            
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
        batchnorm_encode_variables, batchnorm_decode_variables = tsdae_apply_functions.align_batchnorm_variables(batchnorm_variables, d['current_apply_activation_to_embedding'], d['apply_activation_to_output'])
#    recon = {}
#    embed = {}
#    error = {}
#    embed_preactivation = {}
#    for partition in partitions:
#        if d['use_batchnorm']:
#            recon[partition], embed[partition], error[partition] = tsdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
#            embed_preactivation[partition] = tsdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
#        else:
#            recon[partition], embed[partition], error[partition] = tsdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True)
#            embed_preactivation[partition] = tsdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
#        print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)
#        if d['current_dimensions'] == d['all_dimensions'] and (not d['use_finetuning'] or d['current_finetuning_run'] > 0):
#            datasetIO.save_datamatrix('{0}/{1}_embedding_layer{2!s}_finetuning{3!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed[partition])
#            datasetIO.save_datamatrix('{0}/{1}_embedding_layer{2!s}_finetuning{3!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed[partition])
#            if d['current_apply_activation_to_embedding']:
#                datasetIO.save_datamatrix('{0}/{1}_embedding_preactivation_layer{2!s}_finetuning{3!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed_preactivation[partition])
#                datasetIO.save_datamatrix('{0}/{1}_embedding_preactivation_layer{2!s}_finetuning{3!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed_preactivation[partition])
            
    
    
    
    
    # compute embedding and reconstruction
    print('computing embedding and reconstruction...', flush=True)        
    recon = {}
    embed = {}
    error = {}
    embed_preactivation = {}
    for partition in ['all']:
        if np.isnan(dataset[partition].matrix).any():
            print('datamatrix has missing values. random imputation...', flush=True)
            dp = copy.deepcopy(dataset[partition])
            is_missing = np.isnan(dp.matrix)
            for i in range(5):
                print('impute iteration {0!s}'.format(i), flush=True)
                normal_noise = np.random.randn(dp.shape[0], dp.shape[1])*marginalstdvs
                bernoulli_noise = (np.random.rand(dp.shape[0], dp.shape[1]) <= marginalprobabilities).astype('float64')
                noise = bernoulli_noise*isbernoullimarginal + normal_noise*(1-isbernoullimarginal)
                dp.matrix[is_missing] = noise[is_missing]
                if i == 0:
                    if d['use_batchnorm']:
                        recon[partition], embed[partition], error[partition] = tsdae_apply_functions.encode_and_decode(dp, W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                        if d['current_apply_activation_to_embedding']:
                            embed_preactivation[partition] = tsdae_apply_functions.encode(dp, W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
                    else:
                        recon[partition], embed[partition], error[partition] = tsdae_apply_functions.encode_and_decode(dp, W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True)
                        if d['current_apply_activation_to_embedding']:
                            embed_preactivation[partition] = tsdae_apply_functions.encode(dp, W, Be, activation_function['np'], apply_activation_to_embedding=False)
                else:
                    if d['use_batchnorm']:
                        reconi, embedi, errori = tsdae_apply_functions.encode_and_decode(dp, W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                        if d['current_apply_activation_to_embedding']:
                            embed_preactivationi = tsdae_apply_functions.encode(dp, W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
                    else:
                        reconi, embedi, errori = tsdae_apply_functions.encode_and_decode(dp, W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True)
                        if d['current_apply_activation_to_embedding']:
                            embed_preactivationi = tsdae_apply_functions.encode(dp, W, Be, activation_function['np'], apply_activation_to_embedding=False)
                    recon[partition].matrix += reconi.matrix
                    embed[partition].matrix += embedi.matrix
                    error[partition] += errori
                    if d['current_apply_activation_to_embedding']:
                        embed_preactivation[partition].matrix += embed_preactivationi.matrix
            recon[partition].matrix /= 5
            embed[partition].matrix /= 5
            error[partition] /= 5
            if d['current_apply_activation_to_embedding']:
                embed_preactivation[partition].matrix /= 5
        else:
            if d['use_batchnorm']:
                recon[partition], embed[partition], error[partition] = tsdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                if d['current_apply_activation_to_embedding']:
                    embed_preactivation[partition] = tsdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
            else:
                recon[partition], embed[partition], error[partition] = tsdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], dataset['train'].columnmeta['likelihood'] == 'bernoulli', return_embedding=True, return_reconstruction_error=True)
                if d['current_apply_activation_to_embedding']:
                    embed_preactivation[partition] = tsdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
        print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)
    if d['current_dimensions'] == d['all_dimensions'] and (not d['use_finetuning'] or d['current_finetuning_run'] > 0):
        for partition in partitions:
            recon[partition] = recon['all'].tolabels(rowlabels=dataset[partition].rowlabels.copy())
            embed[partition] = embed['all'].tolabels(rowlabels=dataset[partition].rowlabels.copy())
            if d['current_apply_activation_to_embedding']:
                embed_preactivation[partition] = embed_preactivation['all'].tolabels(rowlabels=dataset[partition].rowlabels.copy())
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
    
    fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
    ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
    ax.loglog(reporting_steps, train_losses, ':r', linewidth=1, label='train')
    ax.loglog(reporting_steps, valid_losses, '-g', linewidth=1, label='valid')
    ax.loglog(reporting_steps, train_noisy_losses, '--b', linewidth=1, label='train,noisy')
    ax.loglog(reporting_steps, valid_noisy_losses, '-.k', linewidth=1, label='valid,noisy')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylabel('loss', fontsize=8)
    ax.set_xlabel('steps (selected step:{0!s})'.format(selected_step), fontsize=8)
    ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
    # ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', left=True, right=True, bottom=True, top=False,
                   labelleft=True, labelright=False, labelbottom=True, labeltop=False, labelsize=8)
    fg.savefig('{0}/loglog_optimization_path_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
    plt.close()
        
        
        
        
#    # PLOT RECONSTRUCTIONS
#    print('plotting reconstructions...', flush=True)
#    num_recons = min([d['reconstruction_rows']*d['reconstruction_cols'], dataset['valid'].shape[0]])
#    x_valid = dataset['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] != 'bernoulli']
#    xr_valid = recon['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] != 'bernoulli']
#    if x_valid.shape[1] > 1000:
#        x_valid = x_valid[:,:1000]
#        xr_valid = xr_valid[:,:1000]
#    lb = np.append(x_valid, xr_valid, 1).min(1)
#    ub = np.append(x_valid, xr_valid, 1).max(1)
#    fg, axs = plt.subplots(2*d['reconstruction_rows'], d['reconstruction_cols'], figsize=(6.5,6.5))
#    for i, ax in enumerate(axs.reshape(-1)[:d['reconstruction_rows']*d['reconstruction_cols']]):
#        if i < num_recons:
#            ax.plot(x_valid[i,:], xr_valid[i,:], 'ok', markersize=0.5, markeredgewidth=0, alpha=0.1)
#            ax.set_ylim(lb[i], ub[i])
#            ax.set_xlim(lb[i], ub[i])
#            ax.tick_params(axis='both', which='major', left=False, right=False, bottom=False, top=False, labelleft=False, labelright=False, labelbottom=False, labeltop=False, pad=4)
#            ax.set_frame_on(False)
#            ax.axvline(lb[i], linewidth=1, color='k')
#            ax.axvline(ub[i], linewidth=1, color='k')
#            ax.axhline(lb[i], linewidth=1, color='k')
#            ax.axhline(ub[i], linewidth=1, color='k')
#        else:
#            fg.delaxes(ax)
#    x_valid = dataset['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] == 'bernoulli']
#    xr_valid = recon['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] == 'bernoulli']
#    if x_valid.shape[1] > 1000:
#        x_valid = x_valid[:,:1000]
#        xr_valid = xr_valid[:,:1000]
#    x_valid = x_valid.astype('bool')
#    lb = -0.05
#    ub = 1.05
#    for i, ax in enumerate(axs.reshape(-1)[d['reconstruction_rows']*d['reconstruction_cols']:]):
#        if i < num_recons:
#            ax.boxplot([xr_valid[i,~x_valid[i,:]], xr_valid[i,x_valid[i,:]]], positions=[0.2, 0.8])
#            ax.set_ylim(lb, ub)
#            ax.set_xlim(lb, ub)
#            ax.tick_params(axis='both', which='major', left=False, right=False, bottom=False, top=False, labelleft=False, labelright=False, labelbottom=False, labeltop=False, pad=4)
#            ax.set_frame_on(False)
#            ax.axvline(lb, linewidth=1, color='k')
#            ax.axvline(ub, linewidth=1, color='k')
#            ax.axhline(lb, linewidth=1, color='k')
#            ax.axhline(ub, linewidth=1, color='k')
#        else:
#            fg.delaxes(ax)
#    fg.savefig('{0}/reconstructions_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=1200)
#    plt.close()
    
    
    
    
    
    
    
    # plot reconstructions
    print('plotting reconstructions...', flush=True)
    dataset['valid'].matrix[dataset['valid'].matrix == nanvalue] = np.nan
    num_recons = min([d['reconstruction_rows']*d['reconstruction_cols'], dataset['valid'].shape[0]])
    x_valid = dataset['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] != 'bernoulli']
    xr_valid = recon['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] != 'bernoulli']
    if x_valid.shape[1] > 1000:
        x_valid = x_valid[:,:1000]
        xr_valid = xr_valid[:,:1000]
    lb = np.nanmin(np.append(x_valid, xr_valid, 1), 1)
    ub = np.nanmax(np.append(x_valid, xr_valid, 1), 1)
    fg, axs = plt.subplots(2*d['reconstruction_rows'], d['reconstruction_cols'], figsize=(6.5,6.5))
    for i, ax in enumerate(axs.reshape(-1)[:d['reconstruction_rows']*d['reconstruction_cols']]):
        hit = np.logical_and(np.isfinite(x_valid[i,:]), np.isfinite(xr_valid[i,:]))
        if i < num_recons and hit.any():
            ax.plot(x_valid[i,hit], xr_valid[i,hit], 'ok', markersize=0.5, markeredgewidth=0, alpha=0.1)
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
    x_valid = dataset['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] == 'bernoulli']
    xr_valid = recon['valid'].matrix[:num_recons,dataset['train'].columnmeta['likelihood'] == 'bernoulli']
    if x_valid.shape[1] > 1000:
        x_valid = x_valid[:,:1000]
        xr_valid = xr_valid[:,:1000]
    lb = -0.1
    ub = 1.1
    for i, ax in enumerate(axs.reshape(-1)[d['reconstruction_rows']*d['reconstruction_cols']:]):
        hit = np.logical_and(np.isfinite(x_valid[i,:]), np.isfinite(xr_valid[i,:]))
        if i < num_recons and hit.any():
            ax.boxplot([xr_valid[i,x_valid[i,:] == 0], xr_valid[i,x_valid[i,:] == 1]], positions=[0.2, 0.8], flierprops={'markersize':0.5, 'markeredgewidth':0, 'alpha':0.1}, boxprops={'linewidth':0.5}, whiskerprops={'linewidth':0.5}, medianprops={'linewidth':0.5})
            ax.set_ylim(lb, ub)
            ax.set_xlim(lb, ub)
            ax.tick_params(axis='both', which='major', left=False, right=False, bottom=False, top=False, labelleft=False, labelright=False, labelbottom=False, labeltop=False, pad=4)
            ax.set_frame_on(False)
            ax.axvline(lb, linewidth=1, color='k')
            ax.axvline(ub, linewidth=1, color='k')
            ax.axhline(lb, linewidth=1, color='k')
            ax.axhline(ub, linewidth=1, color='k')
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
