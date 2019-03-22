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


# fine tuning run
finetuning_run = 4


# load the data
with open('data/prepared_data/train.pickle', 'rb') as fr:
    train = pickle.load(fr)
with open('data/prepared_data/valid.pickle', 'rb') as fr:
    valid = pickle.load(fr)
train_examples = train.shape[0]
valid_examples = valid.shape[0]


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
dimensions = [round(input_dimension*x) for x in [1.0, 64.0, 16.0, 4.0, 1.0, 1.0/4.0, 1.0/16.0]]
hiddenlayers = len(dimensions) - 1
noise_probability = 1.0
noise_sigma = 0.9/2.0
noise_distribution = sps.truncnorm(-2.0, 2.0, scale=noise_sigma)
reports_per_log = 10 # 20
initialization_sigma = 0.1
initialization_distribution = tf.truncated_normal


# define placeholders, use None to allow feeding different numbers of examples
x = tf.placeholder(tf.float32, [None, input_dimension])
noise = tf.placeholder(tf.float32, [None, input_dimension])
noise_mask = tf.placeholder(tf.float32, [None, input_dimension])


# define variables
global_step, W, bencode, bdecode = create_variables(dimensions, initialization_distribution, initialization_sigma)


# update variables
variables_path = 'results/autoencoder/variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run-1)
fix_or_init = 'init'
include_global_step = False # False for first finetuning
if os.path.exists(variables_path):
    global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode), variables_path, fix_or_init, include_global_step)
else:
    raise ValueError('no prior weights found!')


# define variables initializer
#init = tf.global_variables_initializer()


# define model
activation_function = tf.tanh # tf.nn.elu, tf.sigmoid, tf.tanh
apply_activation_to_output = False
h, hhat, xhat = create_autoencoder(x, activation_function, apply_activation_to_output, W, bencode, bdecode)
#xnoisy = x + noise_mask*(noise - x)
xnoisy = x + noise_mask*noise
hnoisy, hhatnoisy, xhatnoisy = create_autoencoder(xnoisy, activation_function, apply_activation_to_output, W, bencode, bdecode)


# define loss
loss = tf.reduce_mean(tf.squared_difference(x, xhat))
noisy_loss = tf.reduce_mean(tf.squared_difference(x, xhatnoisy))


# start tensorflow session
sess = tf.InteractiveSession()


# define optimizer and training function
learning_rate = 0.001
epsilon = 0.001
beta1 = 0.9
beta2 = 0.999
#learning_rate = 0.001 # defaults
#epsilon = 10**-8
#beta1 = 0.9
#beta2 = 0.999
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
train_fn = optimizer.minimize(noisy_loss, global_step=global_step)
saver = tf.train.Saver()


# initialize session
init = tf.global_variables_initializer()
sess.run(init)


# collect losses and visualizations for checkpoints
reporting_steps = np.concatenate((np.arange(200000, 1800000, 100000, dtype='int32'), np.arange(1900000, 2400000, 100000, dtype='int32'), np.arange(4500000, 5100000, 100000, dtype='int32'), np.arange(10800000, 11300000, 100000, dtype='int32'), np.arange(11700000, 12200000, 100000, dtype='int32'), np.arange(12600000, 13100000, 100000, dtype='int32'), np.arange(13900000, 14400000, 100000, dtype='int32'), np.arange(16500000, 17000000, 100000, dtype='int32'), np.arange(18300000, 18800000, 100000, dtype='int32'), np.arange(19300000, 19800000, 100000, dtype='int32')))
#reporting_steps = np.array([636581, 1497131], dtype='int32') # np.arange(200000, 900000, 100000, dtype='int32')
plotting_steps = {500000, 1000000, 1500000, 5000000, 10800000, 12000000, 13000000, 14300000, 16900000, 18700000, 19700000}
valid_losses = np.zeros(reporting_steps.size, dtype='float32')
train_losses = np.zeros(reporting_steps.size, dtype='float32')
valid_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
train_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
for i, reporting_step in enumerate(reporting_steps):
    saver.restore(sess, 'results/autoencoder/intermediate_variables_layer{0!s}_finetuning{1!s}_step{2!s}'.format(hiddenlayers, finetuning_run, reporting_step))
    valid_losses[i] = sess.run(loss, feed_dict={x:valid.matrix})
    train_losses[i] = sess.run(loss, feed_dict={x:train.matrix})
    train_noisy_losses[i] = sess.run(noisy_loss, feed_dict={x:train.matrix, noise_mask:(np.random.rand(train.shape[0], input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=train.shape[0]*input_dimension).reshape(train.shape[0], input_dimension)).astype('float32')})
    valid_noisy_losses[i] = sess.run(noisy_loss, feed_dict={x:valid.matrix, noise_mask:(np.random.rand(valid.shape[0], input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=valid.shape[0]*input_dimension).reshape(valid.shape[0], input_dimension)).astype('float32')})

    if reporting_step in plotting_steps:
        # view 2d projection
        proj2d_train = sess.run(h[-1], feed_dict={x:train.matrix})
        proj2d_valid = sess.run(h[-1], feed_dict={x:valid.matrix})
        proj2d_train_preactivation = 0.5*np.log((1+proj2d_train)/(1-proj2d_train))
        proj2d_valid_preactivation = 0.5*np.log((1+proj2d_valid)/(1-proj2d_valid))
        fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
        ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
        ax.plot(proj2d_train[:,0], proj2d_train[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
        ax.plot(proj2d_valid[:,0], proj2d_valid[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
        ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
        ax.set_frame_on(False)
        #fg.savefig('results/autoencoder/proj2d_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
        fg.show()
    #    fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
    #    ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
    #    ax.plot(proj2d_train_preactivation[:,0], proj2d_train_preactivation[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
    #    ax.plot(proj2d_valid_preactivation[:,0], proj2d_valid_preactivation[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
    #    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
    #    ax.set_frame_on(False)
    #    #fg.savefig('results/autoencoder/proj2d_preactivation_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
    #    fg.show()


# plot loss
fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
ax.semilogx(reporting_steps, train_losses, ':k', linewidth=1, label='train')
ax.semilogx(reporting_steps, valid_losses, '-k', linewidth=1, label='valid')
ax.semilogx(reporting_steps, train_noisy_losses, ':r', linewidth=1, label='train, noisy')
ax.semilogx(reporting_steps, valid_noisy_losses, '-r', linewidth=1, label='valid, noisy')
ax.legend(loc='best', fontsize=8)
ax.set_ylabel('loss', fontsize=8, fontname='arial')
ax.set_xlabel('steps', fontsize=8, fontname='arial')
ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
ax.set_ylim(0, 1)
ax.tick_params(axis='both', which='major', left='on', right='on', bottom='on', top='off', labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
fg.savefig('results/autoencoder/optimization_path_layer{0!s}_finetuning{1!s}_step{2!s}.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
fg.show()


# view 100 reconstructions
x_valid = valid.matrix[:7*14,:]
xr_valid = sess.run(xhat, feed_dict={x:x_valid})
lb = np.append(x_valid, xr_valid, 1).min(1)
ub = np.append(x_valid, xr_valid, 1).max(1)
fg, axs = plt.subplots(7, 14, figsize=(6.5,3.25))
for i, ax in enumerate(axs.reshape(-1)):
    ax.plot(x_valid[i,:], xr_valid[i,:], 'ok', markersize=2, markeredgewidth=0)
    ax.set_ylim(lb[i], ub[i])
    ax.set_xlim(lb[i], ub[i])
    ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
    ax.set_frame_on(False)
    ax.axvline(lb[i], linewidth=1, color='k')
    ax.axvline(ub[i], linewidth=1, color='k')
    ax.axhline(lb[i], linewidth=1, color='k')
    ax.axhline(ub[i], linewidth=1, color='k')
fg.savefig('results/autoencoder/reconstructions_layer{0!s}_finetuning{1!s}_step{2!s}.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
fg.show()


# view 2d projection
proj2d_train = sess.run(h[-1], feed_dict={x:train.matrix})
proj2d_valid = sess.run(h[-1], feed_dict={x:valid.matrix})
proj2d_train_preactivation = 0.5*np.log((1+proj2d_train)/(1-proj2d_train))
proj2d_valid_preactivation = 0.5*np.log((1+proj2d_valid)/(1-proj2d_valid))
fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
ax.plot(proj2d_train[:,0], proj2d_train[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.plot(proj2d_valid[:,0], proj2d_valid[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/proj2d_layer{0!s}_finetuning{1!s}_step{2!s}.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
fg.show()
fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
ax.plot(proj2d_train_preactivation[:,0], proj2d_train_preactivation[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.plot(proj2d_valid_preactivation[:,0], proj2d_valid_preactivation[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/proj2d_preactivation_layer{0!s}_finetuning{1!s}_step{2!s}.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
fg.show()


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

fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
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
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_step{2!s}_coloredbytissueenrichment.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
fg.show()

#fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
#ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
#for tissue, color in zip(tissues, colors):
#    if tissue == 'none':
#        zorder = 0
#        alpha = 0.05
#        color = 'k'
#    else:
#        zorder = 1
#        alpha = 0.5
#    hitt = train.rowmeta['Tissue'] == tissue
#    hitv = valid.rowmeta['Tissue'] == tissue
#    ax.plot(np.append(proj2d_train[hitt,0], proj2d_valid[hitv,0], 0), np.append(proj2d_train[hitt,1], proj2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
#ax.set_xlim(-0.2, 0.3)
#ax.set_ylim(-0.2, 0.3)
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
#ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
#ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_step{2!s}_coloredbytissueenrichment_zoom.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
#fg.show()

fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
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
ax.set_xlim(np.percentile(np.append(proj2d_train_preactivation[:,0], proj2d_valid_preactivation[:,0], 0), 0.05), np.percentile(np.append(proj2d_train_preactivation[:,0], proj2d_valid_preactivation[:,0], 0), 99.95))
ax.set_ylim(np.percentile(np.append(proj2d_train_preactivation[:,1], proj2d_valid_preactivation[:,1], 0), 0.05), np.percentile(np.append(proj2d_train_preactivation[:,0], proj2d_valid_preactivation[:,0], 0), 99.95))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/proj2d_0-1_preactivation_layer{0!s}_finetuning{1!s}_step{2!s}_coloredbytissueenrichment.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
fg.show()

#fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
#ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
#for tissue, color in zip(tissues, colors):
#    if tissue == 'none':
#        zorder = 0
#        alpha = 0.05
#        color = 'k'
#    else:
#        zorder = 1
#        alpha = 0.5
#    hitt = train.rowmeta['Tissue'] == tissue
#    hitv = valid.rowmeta['Tissue'] == tissue
#    ax.plot(np.append(proj2d_train_preactivation[hitt,0], proj2d_valid_preactivation[hitv,0], 0), np.append(proj2d_train_preactivation[hitt,1], proj2d_valid_preactivation[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
#ax.set_xlim(-0.2, 0.3)
#ax.set_ylim(-0.2, 0.3)
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
#ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
#ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_preactivation_layer{0!s}_finetuning{1!s}_step{2!s}_coloredbytissueenrichment_zoom.png'.format(hiddenlayers, finetuning_run, reporting_step), transparent=True, pad_inches=0, dpi=600)
#fg.show()


'''
# end tensorflow session
sess.close()
# close figures
for i in range(24):
    plt.close()
# clear everything
%reset -f
'''


'''
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
    fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
    ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
    ax.plot(np.append(proj2d_train[hitt,0], proj2d_valid[hitv,0], 0), np.append(proj2d_train[hitt,1], proj2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
    ax.set_frame_on(False)
    fg.show()
'''
