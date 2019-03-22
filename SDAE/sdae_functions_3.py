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

def apply_noise(x, noise, noise_mask, noise_operation):
    # returns the input corrupted by noise
    if noise_operation == 'add':
        return x + noise_mask*noise
    elif noise_operation == 'replace':
        return x + noise_mask*(noise - x)
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
    elif d['noise_distribution'] == 'uniform':
        noise_distribution = tf.random_uniform
        
    # specify distribution of initial weights
    if d['initialization_distribution'] == 'truncnorm':
        initialization_distribution = tf.truncated_normal
    
    # specify activation function
    if d['activation_function'] == 'tanh':
        activation_function = tf.tanh
    elif d['activation_function'] == 'relu':
        activation_function = tf.nn.relu
    elif d['activation_function'] == 'elu':
        activation_function = tf.nn.elu
    elif d['activation_function'] == 'sigmoid':
        activation_function = tf.sigmoid

    # load training data
    with open('{0}/train.pickle'.format(d['input_path']), 'rb') as fr:
        train = pickle.load(fr)
    d['train_examples'] = train.shape[0]
    
    # load validation data
    with open('{0}/valid.pickle'.format(d['input_path']), 'rb') as fr:
        valid = pickle.load(fr)
    d['valid_examples'] = valid.shape[0]
    
    # create output directory
    if not os.path.exists(d['output_path']):
        os.makedirs(d['output_path'])
            
    # initialize model architecture (number of layers and dimension of each layer)
    d['current_dimensions'] = d['all_dimensions'][:d['current_hidden_layer']+1] # dimensions of model up to current depth
    
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
    noise_stdv = tf.placeholder(tf.float32, [])
    noise_prob = tf.placeholder(tf.float32, [])
    training_and_validation_data_initializer = tf.placeholder(tf.float32, [train.shape[0]+valid.shape[0], train.shape[1]])
    selection_mask = tf.placeholder(tf.bool, [train.shape[0]+valid.shape[0]])

    # define variables
    # W contains the weights, bencode contains the biases for encoding, and bdecode contains the biases for decoding
    print('defining variables...', flush=True)
    training_and_validation_data = tf.Variable(training_and_validation_data_initializer, trainable=False, collections=[])
    if os.path.exists(d['previous_variables_path']):
        # update variables (if continuing from a previous training run)
        print('loading previous variables...', flush=True)
        global_step, W, bencode, bdecode = update_variables(d['current_dimensions'], initialization_distribution, d['initialization_sigma'], d['previous_variables_path'], d['fix_or_init'], d['include_global_step'])
    elif d['current_hidden_layer'] == 1 and d['current_finetuning_run'] == 0:
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
    if d['noise_distribution'] == 'truncnorm':
        noise = noise_distribution(tf.shape(x), stddev=noise_stdv)
    else:
        noise = noise_distribution(tf.shape(x), minval=-noise_stdv, maxval=noise_stdv)
    noise_mask = tf.to_float(tf.random_uniform(tf.shape(x)) <= noise_prob)
    xnoisy = apply_noise(x, noise, noise_mask, d['noise_operation'])
    h, hhat, xhat = create_autoencoder(xnoisy, activation_function, d['apply_activation_to_output'], W, bencode, bdecode)
    
    # define loss
    print('defining loss...', flush=True)
    loss = tf.reduce_mean(tf.squared_difference(x, xhat)) # squared error loss

    # define optimizer and training function
    print('defining optimizer and training function...', flush=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=d['learning_rate'], epsilon=d['epsilon'], beta1=d['beta1'], beta2=d['beta2'])
    train_fn = optimizer.minimize(loss, global_step=global_step)

    # define bottleneck layer preactivation
    bottleneck_preactivation = tf.matmul(h[-2], W[-1]) + bencode[-1]


        
        
    # INITIALIZE TENSORFLOW SESSION
    print('initializing tensorflow session...', flush=True)
    init = tf.global_variables_initializer()
    session_config = configure_session(d['processor'], d['gpu_memory_fraction'])
    with tf.Session(config=session_config) as sess:
        sess.run(init)
       
        

        
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
                    sess.run(train_fn, feed_dict={selection_mask:selected, noise_prob:d['noise_probability'], noise_stdv:d['noise_sigma']})
                    
                    # record training and validation errors
                    if training_step == reporting_steps[i]:
                        train_losses[i] = sess.run(loss, feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0})
                        train_noisy_losses[i] = sess.run(loss, feed_dict={selection_mask:is_train, noise_prob:d['noise_probability'], noise_stdv:d['noise_sigma']})
                        valid_losses[i] = sess.run(loss, feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0})
                        valid_noisy_losses[i] = sess.run(loss, feed_dict={selection_mask:is_valid, noise_prob:d['noise_probability'], noise_stdv:d['noise_sigma']})
                        print('step:{0:1.6g}, train loss:{1:1.3g}, valid loss:{2:1.3g}, train noisy loss:{3:1.3g},valid noisy loss:{4:1.3g}, time:{5:1.6g}'.format(reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], time.time() - starttime), flush=True)
                        fl.write('\t'.join(['{0:1.6g}'.format(x) for x in [reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], time.time() - starttime]]) + '\n')
                            
                        # save current weights, reconstructions, and projections
                        if training_step >= d['startsavingstep'] or training_step == reporting_steps[-1]:
                            with open('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], training_step), 'wb') as fw:
                                pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)
                            with open('{0}/intermediate_reconstructions_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], training_step), 'wb') as fw:
                                pickle.dump((sess.run(xhat, feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0}), # train_reconstructed
                                             sess.run(xhat, feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0})), fw) # valid_reconstructed
                            if d['current_dimensions'][-1] == 2:
                                with open('{0}/intermediate_projections_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], training_step), 'wb') as fw:
                                    pickle.dump((sess.run(h[-1], feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0}), # proj2d_train
                                                 sess.run(h[-1], feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0}), # proj2d_valid
                                                 sess.run(bottleneck_preactivation, feed_dict={selection_mask:is_train, noise_prob:0, noise_stdv:0}), # proj2d_train_preactivation
                                                 sess.run(bottleneck_preactivation, feed_dict={selection_mask:is_valid, noise_prob:0, noise_stdv:0})), fw) # proj2d_valid_preactivation
                            
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
        selected_step = max([reporting_steps[i-d['overfitting_score_max']], d['startsavingstep']])
        selected_file_operation = shutil.copyfile
    else:
        print('completed all training steps...', flush=True)
        selected_step = reporting_steps[-1]
        selected_file_operation = shutil.move
    print('selected step:{0}...'.format(selected_step), flush=True)
    
    
    
    
    # SAVE RESULTS
    print('saving results...', flush=True)
    with open('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'wb') as fw:
        pickle.dump({'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses, 'valid_noisy_losses':valid_noisy_losses, 'train_noisy_losses':train_noisy_losses}, fw)
    selected_file_operation('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                            '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
    selected_file_operation('{0}/intermediate_reconstructions_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                            '{0}/reconstructions_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
    if d['current_dimensions'][-1] == 2:
        selected_file_operation('{0}/intermediate_projections_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                                '{0}/projections_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))

    
    
        
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
    # ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', left='on', right='on', bottom='on', top='off',
                   labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
    fg.savefig('{0}/optimization_path_layer{1!s}_finetuning{2!s}.png'\
               .format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
    plt.close()
        
        
        
        
    # PLOT RECONSTRUCTIONS
    print('plotting reconstructions...', flush=True)
    with open('{0}/reconstructions_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
        xr_valid = pickle.load(fr)[1][:d['reconstruction_rows']*d['reconstruction_cols'],:]
    x_valid = valid.matrix[:d['reconstruction_rows']*d['reconstruction_cols'],:]
    if x_valid.shape[1] > 1000:
        x_valid = x_valid[:,:1000]
        xr_valid = xr_valid[:,:1000]
    lb = np.append(x_valid, xr_valid, 1).min(1)
    ub = np.append(x_valid, xr_valid, 1).max(1)
    fg, axs = plt.subplots(d['reconstruction_rows'], d['reconstruction_cols'], figsize=(6.5,3.25))
    for i, ax in enumerate(axs.reshape(-1)):
        ax.plot(x_valid[i,:], xr_valid[i,:], 'ok', markersize=0.5, markeredgewidth=0)
        ax.set_ylim(lb[i], ub[i])
        ax.set_xlim(lb[i], ub[i])
        ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
        ax.set_frame_on(False)
        ax.axvline(lb[i], linewidth=1, color='k')
        ax.axvline(ub[i], linewidth=1, color='k')
        ax.axhline(lb[i], linewidth=1, color='k')
        ax.axhline(ub[i], linewidth=1, color='k')
    fg.savefig('{0}/reconstructions_layer{1!s}_finetuning{2!s}.png'\
               .format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=1200)
    plt.close()
    
    
    
    
    # PLOT 2D PROJECTION
    if d['current_dimensions'][-1] == 2:
        print('plotting 2d projections...', flush=True)
        with open('{0}/projections_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
            proj2d_train, proj2d_valid, proj2d_train_preactivation, proj2d_valid_preactivation = pickle.load(fr)
        
        fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
        ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
        ax.plot(proj2d_train[:,0], proj2d_train[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
        ax.plot(proj2d_valid[:,0], proj2d_valid[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
        ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off',
                       left='off', right='off', labelleft='off', labelright='off', pad=4)
        ax.set_frame_on(False)
        fg.savefig('{0}/proj2d_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
        plt.close()
        
        fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
        ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
        ax.plot(proj2d_train_preactivation[:,0], proj2d_train_preactivation[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
        ax.plot(proj2d_valid_preactivation[:,0], proj2d_valid_preactivation[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
        ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
        ax.set_frame_on(False)
        fg.savefig('{0}/proj2d_preactivation_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
        plt.close()
        
        
        
        
    print('done training phase.', flush=True)

    return d['current_hidden_layer'], d['current_finetuning_run'], d['current_epochs']
