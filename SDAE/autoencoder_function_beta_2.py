# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
custompaths = ['../pkgs']
#custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
#               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
#               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
#               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import scipy.stats as sps


def get_all_dimensions(input_dimension, layer_scaling_factor, min_dimension, first_hidden_layer_heuristic):
    # returns a list of layer dimensions of the encoder, including the input layer
    # heuristic architecture is to make a large first hidden layer = layer_scaling_factor*input_dimension
    # then shrink subsequent hidden layers by 1/layer_scaling_factor until a layer with min_dimension is reached
    if first_hidden_layer_heuristic == 'expand':
        all_dimensions = [round(input_dimension*x) for x in [1.0, layer_scaling_factor, 1.0]]
    elif first_hidden_layer_heuristic == 'iso':
        all_dimensions = [round(input_dimension*x) for x in [1.0, 1.0]]
    elif first_hidden_layer_heuristic == 'contract':
        all_dimensions = [round(input_dimension*x) for x in [1.0]]
    elif type(first_hidden_layer_heuristic) != str:
        all_dimensions = [round(input_dimension*x) for x in [1.0, first_hidden_layer_heuristic]]
        input_dimension = all_dimensions[-1]
    else:
        raise ValueError('invalid first_hidden_layer_heuristic')
    p = 0
    while all_dimensions[-1] >= min_dimension*layer_scaling_factor*np.sqrt(2):
        p += 1
        all_dimensions.append(round(input_dimension/(layer_scaling_factor**p)))
    all_dimensions.append(min_dimension)
    return all_dimensions
        
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

def update_variables(variables, variables_path, fix_or_init, include_global_step):
    # updates W, bencode, and bdecode with values from a previous training run
    # previous values can be fixed for layerwise pretraining or initialized for finetuning
    global_step, W, bencode, bdecode = variables
    with open(variables_path, 'rb') as fr:
        global_step_prev, W_prev, bencode_prev, bdecode_prev = pickle.load(fr)
    if include_global_step:
        global_step = tf.Variable(global_step_prev, trainable=False)
    if fix_or_init == 'fix':
        for i, (w, be, bd) in enumerate(zip(W_prev, bencode_prev, bdecode_prev)):
            W[i] = tf.constant(w, tf.float32, w.shape)
            bencode[i] = tf.constant(be, tf.float32, be.shape)
            bdecode[i] = tf.constant(bd, tf.float32, bd.shape)
    elif fix_or_init == 'init':
        for i, (w, be, bd) in enumerate(zip(W_prev, bencode_prev, bdecode_prev)):
            W[i] = tf.Variable(w)
            bencode[i] = tf.Variable(be)
            bdecode[i] = tf.Variable(bd)
    else:
        raise ValueError('fix_or_init must be fix or init')
    return global_step, W, bencode, bdecode

def create_autoencoder(x, activation_function, apply_activation_to_output, W, bencode, bdecode):
    # returns h, a list of activations from input to bottleneck layer
    # hhat, a list of activations from bottleneck layer to output layer
    # xhat, a reference to the output layer (i.e. the reconstruction)
    h = [x]
    for i, (w, be) in enumerate(zip(W, bencode)):
        h.append(activation_function(tf.matmul(h[i], w) + be))
    hhat = [h[-1]]
    for i, (w, bd) in enumerate(zip(W[::-1], bdecode[::-1])):
        if i == len(W)-1 and not apply_activation_to_output:
            hhat.append(tf.matmul(hhat[i], w, transpose_b=True) + bd)
        else:
            hhat.append(activation_function(tf.matmul(hhat[i], w, transpose_b=True) + bd))
    xhat = hhat[-1]
    return h, hhat, xhat

def create_reporting_steps(steps, firstcheckpoint, maxstepspercheckpoint):
    # returns a numpy array of steps when training error and validation error are computed
    if steps == firstcheckpoint:
        return np.array([steps], dtype='int32')
    else:
        ordersofmagnitude = int(np.floor(np.log10(maxstepspercheckpoint)))
        reporting_steps = np.zeros(0, dtype='int32')
        for orderofmagnitude in range(ordersofmagnitude):
            reporting_steps = np.append(reporting_steps, (10**orderofmagnitude)*np.arange(1, 10, 1, dtype='int32'))
        reporting_steps = np.append(reporting_steps,
                                    np.arange(10**ordersofmagnitude, maxstepspercheckpoint, 10**ordersofmagnitude, dtype='int32'))
        reporting_steps = np.append(reporting_steps, np.arange(maxstepspercheckpoint, steps, maxstepspercheckpoint, dtype='int32'))
        reporting_steps = reporting_steps[np.logical_and(reporting_steps > firstcheckpoint, reporting_steps < steps)]
        reporting_steps = np.concatenate(([firstcheckpoint], reporting_steps, [steps]))
        return reporting_steps

def create_noise_mask(shape, noise_probability):
    # returns a boolean mask selecting a random subset of input variables to be corrupted
    return (np.random.rand(shape[0], shape[1]) <= noise_probability).astype('float32')

def create_noise(shape, noise_distribution):
    # returns the noise for corrupting the input
    return noise_distribution.rvs(size=shape[0]*shape[1]).reshape(shape[0], shape[1]).astype('float32')

def apply_noise(x, noise, noise_mask, noise_operation):
    # returns the input corrupted by noise
    if noise_operation == 'add':
        return x + noise_mask*noise
    elif noise_operation == 'replace':
        return x + noise_mask*(noise - x)
    else:
        raise ValueError('noise operation not specified')

def inverse_activation(activations, activation_function):
    # returns the preactivations for a vector of activations
    if activation_function == tf.sigmoid:
        return np.log(activations/(1-activations))
    elif activation_function == tf.tanh:
        return 0.5*np.log((1+activations)/(1-activations))
    elif activation_function == tf.nn.elu:
        preactivations = activations.copy()
        preactivations[activations < 0] = np.log(activations[activations < 0] + 1)
        return preactivations
    elif activation_function == tf.nn.relu:
        preactivations = activations.copy()
        # can't compute inverse of relu
        return preactivations
    else:
        raise ValueError('no inverse specified for activation function')

def configure_session(processor):
    if processor == 'cpu':
        session_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8) # this prevents hogging CPUs
    elif processor == 'gpu':
        session_config = tf.ConfigProto(allow_soft_placement=True) # can choose any available GPU
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.49 # 0.3 # 0.95 # this prevents hogging the GPU. not sure of the ideal setting.
    else:
        raise ValueError('invalid processor. specify cpu or gpu')
    return session_config




def main(current_hidden_layer,
         current_finetuning_run,
         previous_hidden_layer,
         previous_finetuning_run,
         epochs,
         input_path,
         output_path,
         layer_scaling_factor=5.0,
         min_dimension=2,
         first_hidden_layer_heuristic='expand',
         noise_probability=1.0,
         noise_sigma=0.5,
         noise_shape=sps.truncnorm,
         noise_operation='add',
         initialization_sigma=0.01,
         initialization_distribution=tf.truncated_normal,
         learning_rate=0.001,
         epsilon=0.001,
         beta1=0.9,
         beta2=0.999,
         batch_fraction=0.1,
         firstcheckpoint=1,
         maxstepspercheckpoint=int(1e10),
         startsavingstep=int(1e10),
         use_finetuning=True,
         include_global_step=False,
         overfitting_score_max=5,
         activation_function=tf.tanh,
         apply_activation_to_output=False,
         processor='cpu'):


    
    
    # FINISH CONFIGURATION
    print('finishing configuration...', flush=True)
    
    # load training data
    with open('{0}/train.pickle'.format(input_path), 'rb') as fr:
        train = pickle.load(fr)
    train_examples = train.shape[0]
    
    # load validation data
    with open('{0}/valid.pickle'.format(input_path), 'rb') as fr:
        valid = pickle.load(fr)
    valid_examples = valid.shape[0]
    
    # create output directory
    pathparts = output_path.split('/')
    for i in range(len(pathparts)):
        pathpart = '/'.join(pathparts[:i+1])
        if not os.path.exists(pathpart):
            os.mkdir(pathpart)
            
    # initialize model architecture (number of layers and dimension of each layer)
    input_dimension = train.shape[1]
    all_dimensions = get_all_dimensions(input_dimension, layer_scaling_factor, min_dimension, first_hidden_layer_heuristic) # dimensions of full model
    current_dimensions = all_dimensions[:current_hidden_layer+1] # dimensions of model up to current depth (for layerwise pretraining)
    
    # initialize noise distribution
    noise_distribution = noise_shape(-2.0, 2.0, scale=noise_sigma) if noise_shape == sps.truncnorm\
                                                                   else sps.uniform(-noise_sigma, noise_sigma)
    
    # initialize assignments of training examples to mini-batches and number of training steps for stochastic gradient descent
    batch_size = round(batch_fraction*train_examples)
    batch_ids = create_batch_ids(train_examples, batch_size)
    batches = np.unique(batch_ids).size
    steps = epochs*batches
    
    # specify path to weights from previous training run
#    previous_hidden_layer = current_hidden_layer-1 if current_finetuning_run == 0 else current_hidden_layer
#    previous_finetuning_run = 1 if current_hidden_layer > 2 and use_finetuning and current_finetuning_run == 0 else 0
    previous_variables_path = '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'\
                              .format(output_path, previous_hidden_layer, previous_finetuning_run)
    fix_or_init = 'fix' if current_finetuning_run == 0 else 'init' # fix weights for pretraining, init weights for finetuning
    
    # specify rows and columns of figure showing data reconstructions
    reconstruction_rows = int(np.round(np.sqrt(np.min([100, valid_examples])/2)))
    reconstruction_cols = 2*reconstruction_rows
    
    # print some configuration information
    print('input path: {0}'.format(input_path), flush=True)
    print('output path: {0}'.format(output_path), flush=True)
    print('previous variables path: {0}'.format(previous_variables_path), flush=True)
    print('previous variables fix or init: {0}'.format(fix_or_init), flush=True)
    
    
    
    
    # SAVE CONFIGURATION
    print('saving configuration...', flush=True)
    config_dict = {'current_hidden_layer':current_hidden_layer,
                   'current_finetuning_run':current_finetuning_run,
                   'previous_hidden_layer':previous_hidden_layer,
                   'previous_finetuning_run':previous_finetuning_run,
                   'epochs':epochs,
                   'input_path':input_path,
                   'output_path':output_path,
                   'layer_scaling_factor':layer_scaling_factor,
                   'min_dimension':min_dimension,
                   'first_hidden_layer_heuristic':first_hidden_layer_heuristic,
                   'noise_probability':noise_probability,
                   'noise_sigma':noise_sigma,
                   'noise_shape':noise_shape,
                   'noise_operation':noise_operation,
                   'initialization_sigma':initialization_sigma,
                   'initialization_distribution':initialization_distribution,
                   'learning_rate':learning_rate,
                   'epsilon':epsilon,
                   'beta1':beta1,
                   'beta2':beta2,
                   'batch_fraction':batch_fraction,
                   'firstcheckpoint':firstcheckpoint,
                   'maxstepspercheckpoint':maxstepspercheckpoint,
                   'startsavingstep':startsavingstep,
                   'use_finetuning':use_finetuning,
                   'include_global_step':include_global_step,
                   'overfitting_score_max':overfitting_score_max,
                   'activation_function':activation_function,
                   'apply_activation_to_output':apply_activation_to_output,
                   'processor':processor,
                   'train_examples':train_examples,
                   'valid_examples':valid_examples,
                   'input_dimension':input_dimension,
                   'all_dimensions':all_dimensions,
                   'current_dimensions':current_dimensions,
                   'noise_distribution':noise_distribution,
                   'batch_size':batch_size,
                   'batch_ids':batch_ids,
                   'batches':batches,
                   'steps':steps,
                   'previous_variables_path':previous_variables_path,
                   'fix_or_init':fix_or_init,
                   'reconstruction_rows':reconstruction_rows,
                   'reconstruction_cols':reconstruction_cols}
    with open('{0}/config_layer{1!s}_finetuning{2!s}.pickle'\
              .format(output_path, current_hidden_layer, current_finetuning_run), 'wb') as fw:
        pickle.dump(config_dict, fw)
    with open('{0}/config_layer{1!s}_finetuning{2!s}.txt'\
              .format(output_path, current_hidden_layer, current_finetuning_run), 'wt') as fw:
        for k,v in config_dict.items():
            fw.write('\t'.join([k, str(v)]) + '\n')
    
    
    
    
    # DEFINE REPORTING VARIABLES
    print('defining reporting variables...', flush=True)
    reporting_steps = create_reporting_steps(steps, firstcheckpoint, maxstepspercheckpoint)
    valid_losses = np.zeros(reporting_steps.size, dtype='float32')
    train_losses = np.zeros(reporting_steps.size, dtype='float32')
    valid_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
    train_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
    print('reporting steps:', reporting_steps, flush=True)
    
    
    
    
    # DEFINE COMPUTATIONAL GRAPH
    # with tf.device('/gpu:1'): # unnecessary?
    # define placeholders for input data, use None to allow feeding different numbers of examples
    print('defining placeholders...', flush=True)
#    x = tf.placeholder(tf.float32, [None, input_dimension])
    noise_stdv = tf.placeholder(tf.float32, [])
    noise_prob = tf.placeholder(tf.float32, [])
    training_and_validation_data_initializer = tf.placeholder(tf.float32, [train.shape[0]+valid.shape[0], train.shape[1]])
    selection_mask = tf.placeholder(tf.bool, [train.shape[0]+valid.shape[0]])
#    print(selection_mask)
#    noise = tf.placeholder(tf.float32, [None, input_dimension])
#    noise_mask = tf.placeholder(tf.float32, [None, input_dimension]) # controls the fraction of input variables that are corrupted

    # define variables
    print('defining variables...', flush=True)
    training_and_validation_data = tf.Variable(training_and_validation_data_initializer, trainable=False, collections=[])
    global_step, W, bencode, bdecode = create_variables(current_dimensions, initialization_distribution, initialization_sigma)
    # W contains the weights, bencode contains the biases for encoding, and bdecode contains the biases for decoding

    # update variables (if continuing from a previous training run)
    if os.path.exists(previous_variables_path):
        print('loading previous variables...', flush=True)
        global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode),
                                                            previous_variables_path, fix_or_init, include_global_step)
    elif current_hidden_layer > 1:
        raise ValueError('could not find previous variables')

    # define model
    print('defining model...', flush=True)
#    training_slice = tf.train.slice_input_producer([training_data], num_epochs=epochs)[0]
#    print(training_slice)
#    x = tf.train.batch([training_slice], batch_size=batch_size, enqueue_many=False, dynamic_pad=True, allow_smaller_final_batch=True)
#    print(x)
    x = tf.boolean_mask(training_and_validation_data, selection_mask)
    noise = tf.truncated_normal(tf.shape(x), stddev=noise_stdv)\
            if noise_shape in {sps.truncnorm, 'truncnorm', 'truncated_normal'}\
            else tf.random_uniform(tf.shape(x), minval=-noise_stdv, maxval=noise_stdv)
    noise_mask = tf.to_float(tf.random_uniform(tf.shape(x)) <= noise_prob)
    xnoisy = apply_noise(x, noise, noise_mask, noise_operation)
    h, hhat, xhat = create_autoencoder(xnoisy, activation_function, apply_activation_to_output, W, bencode, bdecode)
    # h contains the activations from input layer to bottleneck layer
    # hhat contains the activations from bottleneck layer to output layer
    # xhat is a reference to the output layer (i.e. the reconstruction)

    # define loss
    print('defining loss...', flush=True)
    loss = tf.reduce_mean(tf.squared_difference(x, xhat)) # squared error loss

    # define optimizer and training function
    print('defining optimizer and training function...', flush=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
    train_fn = optimizer.minimize(loss, global_step=global_step)

    # define bottleneck layer preactivation
    bottleneck_preactivation = tf.matmul(h[-2], W[-1]) + bencode[-1]


        
        
    # INITIALIZE TENSORFLOW SESSION
    print('initializing tensorflow session...', flush=True)
    init = tf.global_variables_initializer()
    session_config = configure_session(processor)
#    coord = tf.train.Coordinator()
#    with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=session_config)) as sess:
    with tf.Session(config=session_config) as sess:
        sess.run(init)
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        
        # TRAINING
        print('training...', flush=True)
        sess.run(training_and_validation_data.initializer, feed_dict={training_and_validation_data_initializer: np.append(train.matrix, valid.matrix, 0)})
        validation_id = -1
        batch_and_validation_ids = np.full(train.shape[0]+valid.shape[0], validation_id, dtype=batch_ids.dtype)
        is_train = np.append(np.ones(train.shape[0], dtype='bool'), np.zeros(valid.shape[0], dtype='bool'))
        is_valid = ~is_train
        training_step = 0
        i = 0
        overfitting_score = 0
        stopearly = False
        starttime = time.time()
        
        with open('{0}/log_layer{1!s}_finetuning{2!s}.txt'\
                  .format(output_path, current_hidden_layer, current_finetuning_run), mode='wt', buffering=1) as fl:
            fl.write('\t'.join(['step', 'train_loss', 'valid_loss', 'train_noisy_loss', 'valid_noisy_loss', 'time']) + '\n')
            
            for epoch in range(epochs):
                if stopearly:
                    break
                # randomize assignment of training examples to batches
                np.random.shuffle(batch_ids)
                batch_and_validation_ids[is_train] = batch_ids
                
                for batch in range(batches):
                    training_step += 1
                    
                    # select mini-batch
                    selected = batch_and_validation_ids == batch
                    
                    # update weights
#                    sess.run(train_fn, feed_dict={x:train.matrix[selected,:],
#                                                  noise_mask:create_noise_mask((selected.sum(), input_dimension), noise_probability),
#                                                  noise:create_noise((selected.sum(), input_dimension), noise_distribution)})
#                    sess.run(train_fn, feed_dict={x:train.matrix[selected,:], noise_prob:noise_probability, noise_stdv:noise_sigma})
                    sess.run(train_fn, feed_dict={selection_mask:selected, noise_prob:noise_probability, noise_stdv:noise_sigma})
                    
                    # record training and validation errors
                    if training_step == reporting_steps[i]:
#                        train_losses[i] = sess.run(loss, feed_dict={x:train.matrix, noise_prob:0, noise_stdv:0})
#                        train_noisy_losses[i] = sess.run(loss, feed_dict={x:train.matrix, noise_prob:noise_probability, noise_stdv:noise_sigma})
#                        valid_losses[i] = sess.run(loss, feed_dict={x:valid.matrix, noise_prob:0, noise_stdv:0})
#                        valid_noisy_losses[i] = sess.run(loss, feed_dict={x:valid.matrix, noise_prob:noise_probability, noise_stdv:noise_sigma})
                        train_losses[i] = sess.run(loss, feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0})
                        train_noisy_losses[i] = sess.run(loss, feed_dict={selection_mask:is_train, noise_prob:noise_probability, noise_stdv:noise_sigma})
                        valid_losses[i] = sess.run(loss, feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0})
                        valid_noisy_losses[i] = sess.run(loss, feed_dict={selection_mask:is_valid, noise_prob:noise_probability, noise_stdv:noise_sigma})
                        print('''step:{0:1.6g}, train loss:{1:1.3g}, valid loss:{2:1.3g}, train noisy loss:{3:1.3g}, 
                                 valid noisy loss:{4:1.3g}, time:{5:1.6g}'''\
                              .format(reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i],
                                      valid_noisy_losses[i], time.time() - starttime), flush=True)
                        fl.write('\t'.join(['{0:1.6g}'.format(x) for x in [reporting_steps[i], train_losses[i],
                                                                           valid_losses[i], train_noisy_losses[i],
                                                                           valid_noisy_losses[i], time.time() - starttime]]) + '\n')
                        
                        # save current weights
                        if training_step >= startsavingstep and training_step < reporting_steps[-1]:
                            with open('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'\
                                      .format(output_path, current_hidden_layer, current_finetuning_run, training_step), 'wb') as fw:
                                pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)
                            
                            # stop early if overfitting
                            if valid_losses[i] >= 1.05*(np.insert(valid_losses[:i], 0, np.inf).min()):
                                overfitting_score += 1
                            else:
                                overfitting_score = 0
                            if overfitting_score == overfitting_score_max:
                                stopearly = True
                                print('stopping early!', flush=True)
                                break
                        i += 1
                        
                        
                        
                        
        # ROLL BACK IF OVERFITTING
        if stopearly:
            reporting_steps = reporting_steps[:i+1]
            train_losses = train_losses[:i+1]
            valid_losses = valid_losses[:i+1]
            train_noisy_losses = train_noisy_losses[:i+1]
            valid_noisy_losses = valid_noisy_losses[:i+1]
            rollback_variables_path = '{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'\
                                      .format(output_path, current_hidden_layer, current_finetuning_run,
                                              max([reporting_steps[i-overfitting_score_max], startsavingstep]))
            print('rolling back to {0}...'.format(rollback_variables_path), flush=True)        
            global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode),
                                                                variables_path=rollback_variables_path,
                                                                fix_or_init='init', include_global_step=True)    
            h, hhat, xhat = create_autoencoder(xnoisy, activation_function, apply_activation_to_output, W, bencode, bdecode)
            loss = tf.reduce_mean(tf.squared_difference(x, xhat))
            train_fn = optimizer.minimize(loss, global_step=global_step)
            bottleneck_preactivation = tf.matmul(h[-2], W[-1]) + bencode[-1]
            init = tf.global_variables_initializer()
            sess.run(init)
        
        
        
        
        # SAVE RESULTS
        print('saving results...', flush=True)
        with open('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'\
                  .format(output_path, current_hidden_layer, current_finetuning_run), 'wb') as fw:
            pickle.dump({'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses,
                         'valid_noisy_losses':valid_noisy_losses, 'train_noisy_losses':train_noisy_losses}, fw)
        with open('{0}/variables_layer{1!s}_finetuning{2!s}.pickle'\
                  .format(output_path, current_hidden_layer, current_finetuning_run), 'wb') as fw:
            pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)
        
        
        
        
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
        ax.set_xlabel('steps', fontsize=8)
        ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', left='on', right='on', bottom='on', top='off',
                       labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
        fg.savefig('{0}/optimization_path_layer{1!s}_finetuning{2!s}.png'\
                   .format(output_path, current_hidden_layer, current_finetuning_run), transparent=True, pad_inches=0, dpi=600)
        plt.close()
        
        
        
        
        # PLOT RECONSTRUCTIONS
        print('plotting reconstructions...', flush=True)
        is_recon = np.zeros(train.shape[0]+valid.shape[0], dtype='bool')
        is_recon[train.shape[0]:train.shape[0]+reconstruction_rows*reconstruction_cols] = True
        x_valid = valid.matrix[:reconstruction_rows*reconstruction_cols,:]
        xr_valid = sess.run(xhat, feed_dict={selection_mask:is_recon, noise_prob:0, noise_stdv:0})
        if x_valid.shape[1] > 1000:
            x_valid = x_valid[:,:1000]
            xr_valid = xr_valid[:,:1000]
        lb = np.append(x_valid, xr_valid, 1).min(1)
        ub = np.append(x_valid, xr_valid, 1).max(1)
        fg, axs = plt.subplots(reconstruction_rows, reconstruction_cols, figsize=(6.5,3.25))
        for i, ax in enumerate(axs.reshape(-1)):
            ax.plot(x_valid[i,:], xr_valid[i,:], 'ok', markersize=0.5, markeredgewidth=0)
            ax.set_ylim(lb[i], ub[i])
            ax.set_xlim(lb[i], ub[i])
            ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off',
                           labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
            ax.set_frame_on(False)
            ax.axvline(lb[i], linewidth=1, color='k')
            ax.axvline(ub[i], linewidth=1, color='k')
            ax.axhline(lb[i], linewidth=1, color='k')
            ax.axhline(ub[i], linewidth=1, color='k')
        fg.savefig('{0}/reconstructions_layer{1!s}_finetuning{2!s}.png'\
                   .format(output_path, current_hidden_layer, current_finetuning_run), transparent=True, pad_inches=0, dpi=1200)
        plt.close()
        
        
        
        
        # PLOT 2D PROJECTION
        if current_dimensions[-1] == 2:
            print('plotting 2d projections...', flush=True)
            proj2d_train = sess.run(h[-1], feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0})
            proj2d_valid = sess.run(h[-1], feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0})
            #proj2d_train_preactivation = inverse_activation(proj2d_train, activation_function)
            #proj2d_valid_preactivation = inverse_activation(proj2d_valid, activation_function)
            proj2d_train_preactivation = sess.run(bottleneck_preactivation, feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0})
            proj2d_valid_preactivation = sess.run(bottleneck_preactivation, feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0})
            
            fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
            ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
            ax.plot(proj2d_train[:,0], proj2d_train[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.plot(proj2d_valid[:,0], proj2d_valid[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
            ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off',
                           left='off', right='off', labelleft='off', labelright='off', pad=4)
            ax.set_frame_on(False)
            fg.savefig('{0}/proj2d_layer{1!s}_finetuning{2!s}.png'\
                       .format(output_path, current_hidden_layer, current_finetuning_run), transparent=True, pad_inches=0, dpi=600)
            plt.close()
            
            fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
            ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
            ax.plot(proj2d_train_preactivation[:,0], proj2d_train_preactivation[:,1],
                    'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.plot(proj2d_valid_preactivation[:,0], proj2d_valid_preactivation[:,1],
                    'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
            ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off',
                           left='off', right='off', labelleft='off', labelright='off', pad=4)
            ax.set_frame_on(False)
            fg.savefig('{0}/proj2d_preactivation_layer{1!s}_finetuning{2!s}.png'\
                       .format(output_path, current_hidden_layer, current_finetuning_run), transparent=True, pad_inches=0, dpi=600)
            plt.close()
        
        
        
        
        # end tensorflow session
        print('closing tensorflow session...', flush=True)
#        coord.request_stop()
#        coord.join(threads)
    
    
    
    
    return current_hidden_layer, current_finetuning_run, epochs
