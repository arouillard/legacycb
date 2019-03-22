# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
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

def create_batch_ids(total_size, batch_size=100, dtype='int32'):
    batches = round(total_size/batch_size)
    batch_limits = np.linspace(0, total_size, batches+1, dtype=dtype)
    batch_ids = np.zeros(total_size, dtype=dtype)
    for i, (lb, ub) in enumerate(zip(batch_limits[:-1], batch_limits[1:])):
        batch_ids[lb:ub] = i
    return batch_ids

def create_variables(dimensions, initialization_distribution, initialization_sigma):
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


# load the data
with open('data/prepared_data/train.pickle', 'rb') as fr:
    train = pickle.load(fr)
with open('data/prepared_data/valid.pickle', 'rb') as fr:
    valid = pickle.load(fr)
train_examples = train.shape[0]
valid_examples = valid.shape[0]

with open('C:/Users/ar988996/Documents/DeepLearning/geneid_tissue_mark.pickle', 'rb') as fr:
    geneid_tissue = pickle.load(fr)
train.rowmeta['Tissue'] = np.full(train.shape[0], 'none', dtype='object')
for i, geneid in enumerate(train.rowmeta['GeneID']):
    if geneid in geneid_tissue:
        train.rowmeta['Tissue'][i] = geneid_tissue[geneid]
valid.rowmeta['Tissue'] = np.full(valid.shape[0], 'none', dtype='object')
for i, geneid in enumerate(valid.rowmeta['GeneID']):
    if geneid in geneid_tissue:
        valid.rowmeta['Tissue'][i] = geneid_tissue[geneid]

#tissues = sorted(list(set(train.rowmeta['Tissue'].tolist()).union(valid.rowmeta['Tissue'].tolist())))
tissues = ['Brain', 'Pituitary', 'Spleen', 'Blood', 'Bone Marrow', 'Testis', 'Liver', 'Kidney', 'Salivary Gland', 'Heart', 'Muscle', 'Pancreas', 'Stomach', 'Small Intestine', 'Colon', 'Fallopian Tube', 'Skin', 'Bladder', 'Cervix Uteri', 'Prostate', 'Esophagus', 'Vagina', 'Adrenal Gland', 'Uterus', 'Ovary', 'Breast', 'Nerve', 'Adipose Tissue', 'Blood Vessel', 'Lung', 'Thyroid', 'none']
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/len(tissues))) for i in range(len(tissues))]


# create output directories
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('results/autoencoder'):
    os.mkdir('results/autoencoder')


# define batches
batch_size = round(0.05*train_examples)
batch_ids = create_batch_ids(train_examples, batch_size)
batches = np.unique(batch_ids).size


# define parameters
input_dimension = train.shape[1]
dimensions = [round(input_dimension*x) for x in [1.0, 16.0, 4.0, 1.0, 1.0/4.0, 1.0/16.0]] # + [2]
hiddenlayers = len(dimensions) - 1
noise_probability = 0.05
noise_sigma = 0.5
noise_distribution = sps.truncnorm(-2.0, 2.0, scale=noise_sigma)
reports_per_log = 100
initialization_sigma = 0.1
initialization_distribution = tf.truncated_normal


# define placeholders, use None to allow feeding different numbers of examples
x = tf.placeholder(tf.float32, [None, input_dimension])
noise = tf.placeholder(tf.float32, [None, input_dimension])
noise_mask = tf.placeholder(tf.float32, [None, input_dimension])


# define variables
global_step, W, bencode, bdecode = create_variables(dimensions, initialization_distribution, initialization_sigma)


# collect losses
finetuning_runs = np.arange(13, dtype='int32')
batchess = np.full(finetuning_runs.size, 20, dtype='int32')
batchess[-1] = 5
epochs = np.ones(finetuning_runs.size, dtype='float32')
valid_losses = np.zeros(finetuning_runs.size, dtype='float32')
train_losses = np.zeros(finetuning_runs.size, dtype='float32')
for i, (finetuning_run, batches) in enumerate(zip(finetuning_runs, batchess)):
    # update variables
    variables_path = 'results/autoencoder/variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run)
    fix_or_init = 'init'
    include_global_step = False # False for first finetuning
    if os.path.exists(variables_path):
        global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode), variables_path, fix_or_init, include_global_step)
    # define model
    activation_function = tf.tanh
    apply_activation_to_output = False
    h, hhat, xhat = create_autoencoder(x, activation_function, apply_activation_to_output, W, bencode, bdecode)
    xnoisy = x + noise_mask*noise
    hnoisy, hhatnoisy, xhatnoisy = create_autoencoder(xnoisy, activation_function, apply_activation_to_output, W, bencode, bdecode)
    # define loss
    loss = tf.reduce_mean(tf.squared_difference(x, xhat))
    noisy_loss = tf.reduce_mean(tf.squared_difference(x, xhatnoisy))
    # start tensorflow session
    sess = tf.InteractiveSession()
    # initialize session
    init = tf.global_variables_initializer()
    sess.run(init)
    # calculate losses
    valid_losses[i] = sess.run(loss, feed_dict={x:valid.matrix})
    train_losses[i] = sess.run(loss, feed_dict={x:train.matrix})
    # get epochs
    if i > 0:
        with open('results/autoencoder/optimization_path_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run), 'rb') as fr:
            run_losses = pickle.load(fr)
        epochs[i] = epochs[i-1] + run_losses['reporting_steps'][-1]/batches
    # get best proj2d
    if i == 6:
        proj2d_train = sess.run(h[-1], feed_dict={x:train.matrix})
        proj2d_valid = sess.run(h[-1], feed_dict={x:valid.matrix})
        proj2d_train_preactivation = 0.5*np.log((1+proj2d_train)/(1-proj2d_train))
        proj2d_valid_preactivation = 0.5*np.log((1+proj2d_valid)/(1-proj2d_valid))
        fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
        ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
        for tissue, color in zip(tissues, colors):
            if tissue == 'none':
                zorder = 0
                alpha = 0.05
                color = 'k'
            else:
                zorder = 1
                alpha = 0.5
            hitt = train.rowmeta['Tissue'] == tissue
            hitv = valid.rowmeta['Tissue'] == tissue
            ax.plot(np.append(proj2d_train[hitt,0], proj2d_valid[hitv,0], 0), np.append(proj2d_train[hitt,1], proj2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
        ax.legend(loc='best', ncol=2, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.1)
        ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
        ax.set_frame_on(False)
        fg.savefig('results/autoencoder/proj2d_layer{0!s}_finetuning{1!s}_coloredbytissueenrichment.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
        fg.show()
        fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
        ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
        for tissue, color in zip(tissues, colors):
            if tissue == 'none':
                zorder = 0
                alpha = 0.05
                color = 'k'
            else:
                zorder = 1
                alpha = 0.5
            hitt = train.rowmeta['Tissue'] == tissue
            hitv = valid.rowmeta['Tissue'] == tissue
            ax.plot(np.append(proj2d_train_preactivation[hitt,0], proj2d_valid_preactivation[hitv,0], 0), np.append(proj2d_train_preactivation[hitt,1], proj2d_valid_preactivation[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
        ax.legend(loc='best', ncol=2, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.1)
        ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
        ax.set_frame_on(False)
        fg.savefig('results/autoencoder/proj2d_preactivation_layer{0!s}_finetuning{1!s}_coloredbytissueenrichment.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
        fg.show()
    # end tensorflow session
    sess.close()


# plot loss
fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
ax.semilogx(epochs, train_losses, ':ok', linewidth=1, markersize=2, label='train')
ax.semilogx(epochs, valid_losses, '-sk', linewidth=1, markersize=2, label='valid')
ax.legend(loc='best', fontsize=8)
ax.set_ylabel('loss', fontsize=8, fontname='arial')
ax.set_xlabel('epochs', fontsize=8, fontname='arial')
ax.set_xlim(epochs[0]-1, epochs[-1]+1)
#ax.set_ylim(0.00003, 3)
ax.tick_params(axis='both', which='major', left='on', right='off', bottom='on', top='off', labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
fg.savefig('results/autoencoder/optimization_path_layer{0!s}_finetuningALL.png'.format(hiddenlayers), transparent=True, pad_inches=0, dpi=600)
fg.show()


from sklearn.manifold import TSNE
T = TSNE().fit_transform(np.append(train.matrix, valid.matrix, 0))

fg, ax = plt.subplots(1, 1, figsize=(9.75,6.5))
ax.set_position([0.15/9.75, 0.15/6.5, 9.1/9.75, 6.2/6.5])
for tissue, color in zip(tissues, colors):
    if tissue == 'none':
        zorder = 0
        alpha = 0.05
        color = 'k'
    else:
        zorder = 1
        alpha = 0.5
    hit = np.append(train.rowmeta['Tissue'] == tissue, valid.rowmeta['Tissue'] == tissue)
    ax.plot(T[hit,0], T[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
ax.set_xlim(T.min(0)[0], 1.5*(T.max(0)[0]-T.min(0)[0])-T.max(0)[0])
#ax.set_ylim(T.min(0)[1], T.max(0)[1]+1*(T.max(0)[1]-T.min(0)[1]))
ax.legend(loc='best', ncol=2, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.1)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/tsne2d_coloredbytissueenrichment.png', transparent=True, pad_inches=0, dpi=600)
fg.show()


# plot mean vs y-component
fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
ax.plot(train.matrix.mean(1), proj2d_train[:,1], 'ok', markersize=1, markeredgecolor='k', label='train')
ax.plot(valid.matrix.mean(1), proj2d_valid[:,1], 'ok', markersize=1, markeredgecolor='k', label='valid')
#ax.plot(valid.matrix.mean(1), proj2d_valid[:,1], 'sr', markersize=1, markeredgecolor='r', label='valid')
#ax.legend(loc='best', fontsize=8)
ax.set_ylabel('Y-Component', fontsize=8, fontname='arial')
ax.set_xlabel('Mean', fontsize=8, fontname='arial')
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 1)
ax.tick_params(axis='both', which='major', left='on', right='off', bottom='on', top='off', labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
fg.savefig('results/autoencoder/mean_vs_Ycomponent_layer{0!s}_finetuning6.png'.format(hiddenlayers), transparent=True, pad_inches=0, dpi=600)
fg.show()

# plot stdv vs y-component
fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
ax.plot(train.matrix.std(1), proj2d_train[:,1], 'ok', markersize=1, markeredgecolor='k', label='train')
ax.plot(valid.matrix.std(1), proj2d_valid[:,1], 'ok', markersize=1, markeredgecolor='k', label='valid')
#ax.plot(valid.matrix.mean(1), proj2d_valid[:,1], 'sr', markersize=1, markeredgecolor='r', label='valid')
#ax.legend(loc='best', fontsize=8)
ax.set_ylabel('Y-Component', fontsize=8, fontname='arial')
ax.set_xlabel('Stdv', fontsize=8, fontname='arial')
#ax.set_xlim(-3, 3)
#ax.set_ylim(-1, 1)
ax.tick_params(axis='both', which='major', left='on', right='off', bottom='on', top='off', labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
fg.savefig('results/autoencoder/stdv_vs_Ycomponent_layer{0!s}_finetuning6.png'.format(hiddenlayers), transparent=True, pad_inches=0, dpi=600)
fg.show()
