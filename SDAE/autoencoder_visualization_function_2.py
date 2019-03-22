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
import copy
from machinelearning import datasetIO, dataclasses


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
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.75 # this prevents hogging the GPU. not sure of the ideal setting.
    else:
        raise ValueError('invalid processor. specify cpu or gpu')
    return session_config

def create_embedding_datamatrix(dm, n_components):
    em = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=dm.rowmeta.copy(),
                                columnname='latent_component',
                                columnlabels=np.array(['LC'+str(x) for x in range(n_components)], dtype='object'),
                                columnmeta={},
                                matrixname='sdae_embedding_of_'+dm.matrixname,
                                matrix=np.zeros((dm.shape[0], n_components), dtype='float32'))
    return em




def main(configuration_variables_path, model_variables_path='latest'):

    # LOAD CONFIGURATION VARIABLES
    print('loading configuration variables...', flush=True)
    with open(configuration_variables_path, 'rb') as fr:
        config = pickle.load(fr)
    current_hidden_layer = config['current_hidden_layer']
    current_finetuning_run = config['current_finetuning_run']
    previous_hidden_layer = config['previous_hidden_layer']
    previous_finetuning_run = config['previous_finetuning_run']
    epochs = config['epochs']
    input_path = config['input_path']
    output_path = config['output_path']
    layer_scaling_factor = config['layer_scaling_factor']
    min_dimension = config['min_dimension']
    first_hidden_layer_heuristic = config['first_hidden_layer_heuristic']
    noise_probability = config['noise_probability']
    noise_sigma = config['noise_sigma']
    noise_shape = config['noise_shape']
    noise_operation = config['noise_operation']
    initialization_sigma = config['initialization_sigma']
    initialization_distribution = config['initialization_distribution']
    learning_rate = config['learning_rate']
    epsilon = config['epsilon']
    beta1 = config['beta1']
    beta2 = config['beta2']
    batch_fraction = config['batch_fraction']
    firstcheckpoint = config['firstcheckpoint']
    maxstepspercheckpoint = config['maxstepspercheckpoint']
    startsavingstep = config['startsavingstep']
    use_finetuning = config['use_finetuning']
    include_global_step = config['include_global_step']
    overfitting_score_max = config['overfitting_score_max']
    activation_function = config['activation_function']
    apply_activation_to_output = config['apply_activation_to_output']
    processor = config['processor']
    train_examples = config['train_examples']
    valid_examples = config['valid_examples']
    input_dimension = config['input_dimension']
    all_dimensions = config['all_dimensions']
    current_dimensions = config['current_dimensions']
    noise_distribution = config['noise_distribution']
    batch_size = config['batch_size']
    batch_ids = config['batch_ids']
    batches = config['batches']
    steps = config['steps']
    previous_variables_path = config['previous_variables_path']
    fix_or_init = config['fix_or_init']
    reconstruction_rows = config['reconstruction_rows']
    reconstruction_cols = config['reconstruction_cols']

    # load input data
    partitions = ['train', 'valid', 'test']
    partition_colors = ['k', 'r', 'b']
    dataset = {}
    for partition in partitions:
        dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(input_path, partition))

    # specify path to weights
    if model_variables_path == 'latest':
        model_variables_path = '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'\
                               .format(output_path, current_hidden_layer, current_finetuning_run)
    if not os.path.exists(model_variables_path):
        intermediate_files = [x for x in os.listdir(output_path) if 'intermediate_variables_layer{0!s}_finetuning{1!s}_step'\
                              .format(current_hidden_layer, current_finetuning_run) in x]
        for intermediate_file in intermediate_files:
            print(intermediate_file, flush=True)
        intermediate_steps = [float(x.replace('intermediate_variables_layer{0!s}_finetuning{1!s}_step'\
                              .format(current_hidden_layer, current_finetuning_run), '')\
                              .replace('.pickle','')) for x in intermediate_files]
        for intermediate_step in intermediate_steps:
            print(intermediate_step, flush=True)
        latest_idx = np.argmax(intermediate_steps)
        print(latest_idx, flush=True)
        model_variables_path = '{0}/{1}'.format(output_path, intermediate_files[latest_idx])
    print(model_variables_path, flush=True)




    # LOAD REPORTING VARIABLES
    print('loading reporting variables...', flush=True)
    if os.path.exists('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'\
                      .format(output_path, current_hidden_layer, current_finetuning_run)):
        with open('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'\
                  .format(output_path, current_hidden_layer, current_finetuning_run), 'rb') as fr:
            optimization_path = pickle.load(fr)
        reporting_steps = optimization_path['reporting_steps']
        valid_losses = optimization_path['valid_losses']
        train_losses = optimization_path['train_losses']
        valid_noisy_losses = optimization_path['valid_noisy_losses']
        train_noisy_losses = optimization_path['train_noisy_losses']
    else:
        reporting_steps = np.zeros(0, dtype='int32')
        valid_losses = np.zeros(0, dtype='float32')
        train_losses = np.zeros(0, dtype='float32')
        valid_noisy_losses = np.zeros(0, dtype='float32')
        train_noisy_losses = np.zeros(0, dtype='float32')
        with open('{0}/log_layer{1!s}_finetuning{2!s}.txt'\
                  .format(output_path, current_hidden_layer, current_finetuning_run), 'rt') as fr:
            fr.readline()
            for line in fr:
                step, train_loss, valid_loss, train_noisy_loss, valid_noisy_loss, time = [float(x.strip()) for x in line.split('\t')]
                reporting_steps = np.insert(reporting_steps, reporting_steps.size, step)
                valid_losses = np.insert(valid_losses, valid_losses.size, valid_loss)
                train_losses = np.insert(train_losses, train_losses.size, train_loss)
                valid_noisy_losses = np.insert(valid_noisy_losses, valid_noisy_losses.size, valid_noisy_loss)
                train_noisy_losses = np.insert(train_noisy_losses, train_noisy_losses.size, train_noisy_loss)
                
    for i in range(reporting_steps.size):
        print('''step:{0:1.6g}, train loss:{1:1.3g}, valid loss:{2:1.3g}, 
                 train noisy loss:{3:1.3g}, valid noisy loss:{4:1.3g}'''.format(reporting_steps[i], train_losses[i],
                                                                                valid_losses[i], train_noisy_losses[i],
                                                                                valid_noisy_losses[i]), flush=True)
    
    
    # DEFINE COMPUTATIONAL GRAPH
    # define placeholders for input data, use None to allow feeding different numbers of examples
    print('defining placeholders...', flush=True)
    x = tf.placeholder(tf.float32, [None, input_dimension])
    noise = tf.placeholder(tf.float32, [None, input_dimension])
    noise_mask = tf.placeholder(tf.float32, [None, input_dimension]) # controls the fraction of input variables that are corrupted

    # define variables
    print('defining variables...', flush=True)
    global_step, W, bencode, bdecode = create_variables(current_dimensions, initialization_distribution, initialization_sigma)
    # W contains the weights, bencode contains the biases for encoding, and bdecode contains the biases for decoding

    # load variables
    if os.path.exists(model_variables_path):
        print('loading variables...', flush=True)
        global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode),
                                                            model_variables_path, fix_or_init, include_global_step)
    else:
        raise ValueError('could not find current variables')

    # define model
    print('defining model...', flush=True)
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
    with tf.Session(config=session_config) as sess:
        sess.run(init)
        
        
        
        
        
        
        
        
        
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
        reconstruction = {}
        explained_variance_cumulative_fraction = {}
        reconstruction_mse = {}
        for partition in partitions:
            reconstruction[partition] = copy.deepcopy(dataset[partition])
            reconstruction[partition].matrix = sess.run(xhat, feed_dict={x:dataset[partition].matrix,
                                                        noise_mask:create_noise_mask(dataset[partition].shape, 0),
                                                        noise:create_noise(dataset[partition].shape, noise_distribution)})
            explained_variance_cumulative_fraction[partition] = reconstruction[partition].matrix.var(0).sum()/\
                                                                dataset[partition].matrix.var(0).sum()
            reconstruction_mse[partition] = np.mean((dataset[partition].matrix - reconstruction[partition].matrix)**2)
            print('partition:{0}, expl_var_cumul_frc:{1:1.3g}, recon_mse:{2:1.3g}'.format(partition,
                  explained_variance_cumulative_fraction[partition], reconstruction_mse[partition]), flush=True)

        x_valid = dataset['valid'].matrix[:reconstruction_rows*reconstruction_cols,:]
        xr_valid = reconstruction['valid'].matrix[:reconstruction_rows*reconstruction_cols,:]
        
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
        embedding = {'preactivation':{}, 'activation':{}}
        for partition in partitions:
            embedding['activation'][partition] = create_embedding_datamatrix(dataset[partition], current_dimensions[-1])
            embedding['activation'][partition].matrix = sess.run(h[-1], feed_dict={x:dataset[partition].matrix,
                                                                 noise_mask:create_noise_mask(dataset[partition].shape, 0),
                                                                 noise:create_noise(dataset[partition].shape, noise_distribution)})
            embedding['preactivation'][partition] = create_embedding_datamatrix(dataset[partition], current_dimensions[-1])
            embedding['preactivation'][partition].matrix = sess.run(bottleneck_preactivation, feed_dict={x:dataset[partition].matrix,
                                                                    noise_mask:create_noise_mask(dataset[partition].shape, 0),
                                                                    noise:create_noise(dataset[partition].shape, noise_distribution)})
        if current_dimensions[-1] == 2:
            
            print('plotting 2d projections...', flush=True)
            tissues = ['', 'Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel', 'Brain',
                       'Breast', 'Colon', 'Esophagus', 'Heart', 'Kidney', 'Liver', 'Lung', 'Muscle',
                       'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Salivary Gland', 'Skin',
                       'Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Uterus', 'Vagina']
            tissue_abbrevs = ['X', 'AT', 'AG', 'B', 'BV', 'Bn',
                              'Bt', 'C', 'E', 'H', 'K', 'Lr', 'Lg', 'M',
                              'N', 'O', 'Ps', 'Py', 'Pe', 'SG', 'Sk',
                              'SI', 'Sp', 'St', 'Ts', 'Td', 'U', 'V']
            cmap = plt.get_cmap('gist_rainbow')
            colors = ['k'] + [cmap(float((i+0.5)/(len(tissues)-1))) for i in range(len(tissues)-1)]

            for activation_state in embedding:
                
                for partition in partitions:
                    if 'all' not in embedding[activation_state]:
                        embedding[activation_state]['all'] = copy.deepcopy(embedding[activation_state][partition])
                    else:
                        embedding[activation_state]['all'].append(embedding[activation_state][partition], 0)
                #'''
                fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                for tissue, tissue_abbrev, color in zip(tissues, tissue_abbrevs, colors):
                    if tissue == '-666':
                        continue
                    else:
                        zorder = 1
                        alpha = 0.5
                    hit = embedding[activation_state]['all'].rowmeta['general_tissue'] == tissue
                    hidxs = hit.nonzero()[0]
                    ax.plot(embedding[activation_state]['all'].matrix[hit,0], embedding[activation_state]['all'].matrix[hit,1],
                            linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color,
                            markersize=0.2, markeredgewidth=0, alpha=alpha, zorder=zorder,
                            label='{0}, {1}'.format(tissue_abbrev, tissue))
                    for hidx in hidxs:
                        ax.text(embedding[activation_state]['all'].matrix[hidx,0],
                                embedding[activation_state]['all'].matrix[hidx,1], tissue_abbrev,
                                horizontalalignment='center', verticalalignment='center', fontsize=4, color=color,
                                alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                ax.set_xlim(embedding[activation_state]['all'].matrix[:,0].min(), embedding[activation_state]['all'].matrix[:,0].max())
                ax.set_ylim(embedding[activation_state]['all'].matrix[:,1].min(), embedding[activation_state]['all'].matrix[:,1].max())
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1,
                          numpoints=1, markerscale=40, fontsize=8, labelspacing=0.25)
                ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off',
                               left='off', right='off', labelleft='off', labelright='off', pad=4)
                ax.set_frame_on(False)
                fg.savefig('{0}/proj2d_{1}_layer{2!s}_finetuning{3!s}_coloredby_general_tissue.png'\
                           .format(output_path, activation_state, current_hidden_layer, current_finetuning_run),
                           transparent=True, pad_inches=0, dpi=600)
                plt.close()
                #'''
                                                      
                fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
                ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
                for partition, partition_color in zip(partitions, partition_colors):
                    ax.plot(embedding[activation_state][partition].matrix[:,0],
                            embedding[activation_state][partition].matrix[:,1],
                            'o'+partition_color, markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
                ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off',
                               left='off', right='off', labelleft='off', labelright='off', pad=4)
                ax.set_frame_on(False)
                fg.savefig('{0}/proj2d_{1}_layer{2!s}_finetuning{3!s}.png'\
                           .format(output_path, activation_state, current_hidden_layer, current_finetuning_run),
                           transparent=True, pad_inches=0, dpi=600)
                plt.close()
                
                #'''
                for metadata_label in ['specific_tissue', 'sex', 'age']:
                    metadata_uvals = np.unique(embedding[activation_state]['all'].rowmeta[metadata_label])
                    metadata_colors = [cmap(float((i+0.5)/len(metadata_uvals))) for i in range(len(metadata_uvals))]
                    
                    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                    for metadata_uval, metadata_color in zip(metadata_uvals, metadata_colors):
                        if metadata_uval == '-666':
                            continue
                        else:
                            zorder = 1
                            alpha = 0.5
                        hit = embedding[activation_state]['all'].rowmeta[metadata_label] == metadata_uval
                        hidxs = hit.nonzero()[0]
                        ax.plot(embedding[activation_state]['all'].matrix[hit,0], embedding[activation_state]['all'].matrix[hit,1],
                                linestyle='None', linewidth=0, marker='o', markerfacecolor=metadata_color, markeredgecolor=metadata_color,
                                markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder,
                                label=metadata_uval)
                    ax.set_xlim(embedding[activation_state]['all'].matrix[:,0].min(), embedding[activation_state]['all'].matrix[:,0].max())
                    ax.set_ylim(embedding[activation_state]['all'].matrix[:,1].min(), embedding[activation_state]['all'].matrix[:,1].max())
                    if len(metadata_uvals) > 30:
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1,
                                  numpoints=1, markerscale=2, fontsize=6, labelspacing=0)
                    else:
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1,
                                  numpoints=1, markerscale=4, fontsize=8, labelspacing=0.25)
                    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off',
                                   left='off', right='off', labelleft='off', labelright='off', pad=4)
                    ax.set_frame_on(False)
                    fg.savefig('{0}/proj2d_{1}_layer{2!s}_finetuning{3!s}_coloredby_{4}.png'\
                               .format(output_path, activation_state, current_hidden_layer, current_finetuning_run, metadata_label),
                               transparent=True, pad_inches=0, dpi=600)
                    plt.close()
                #'''
        
        
        
        
        # end tensorflow session
        print('closing tensorflow session...', flush=True)
    
    
    
    
    return configuration_variables_path, model_variables_path

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        raise ValueError('invalid arguments supplied to autoencoder_visualization_function_2.main()')
