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
import copy
import scipy.stats as sps
from machinelearning import datasetselection, featureselection
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score

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
finetuning_run = 6


# load the data
with open('data/prepared_data/train.pickle', 'rb') as fr:
    train = pickle.load(fr)
with open('data/prepared_data/valid.pickle', 'rb') as fr:
    valid = pickle.load(fr)
with open('data/prepared_data/test.pickle', 'rb') as fr:
    test = pickle.load(fr)
train_examples = train.shape[0]
valid_examples = valid.shape[0]
test_examples = test.shape[0]


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
#saver.restore(sess, 'results/autoencoder/variables_layer{0!s}_finetuning{1!s}'.format(hiddenlayers, finetuning_run))
valid_losses = sess.run(loss, feed_dict={x:valid.matrix})
train_losses = sess.run(loss, feed_dict={x:train.matrix})
train_noisy_losses = sess.run(noisy_loss, feed_dict={x:train.matrix, noise_mask:(np.random.rand(train.shape[0], input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=train.shape[0]*input_dimension).reshape(train.shape[0], input_dimension)).astype('float32')})
valid_noisy_losses = sess.run(noisy_loss, feed_dict={x:valid.matrix, noise_mask:(np.random.rand(valid.shape[0], input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=valid.shape[0]*input_dimension).reshape(valid.shape[0], input_dimension)).astype('float32')})


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
#fg.savefig('results/autoencoder/reconstructions_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
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
#fg.savefig('results/autoencoder/proj2d_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()
fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
ax.plot(proj2d_train_preactivation[:,0], proj2d_train_preactivation[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.plot(proj2d_valid_preactivation[:,0], proj2d_valid_preactivation[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_preactivation_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
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
#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_coloredbytissueenrichment.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()

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
ax.set_xlim(-0.2, 0.3)
ax.set_ylim(-0.2, 0.3)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_coloredbytissueenrichment_zoom.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()

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
#fg.savefig('results/autoencoder/proj2d_0-1_preactivation_layer{0!s}_finetuning{1!s}_coloredbytissueenrichment.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()

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
ax.set_xlim(-0.2, 0.3)
ax.set_ylim(-0.2, 0.3)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_preactivation_layer{0!s}_finetuning{1!s}_coloredbytissueenrichment_zoom.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()




raise ValueError('stop')

dm = train
del train
dm.append(valid, 0)
dm.append(test, 0)
print(dm)

recon = copy.deepcopy(dm)
recon.matrix = sess.run(xhat, feed_dict={x:dm.matrix})
recon.matrixname = 'reconstruction_of_' + recon.matrixname
recon.updatedtypeattribute()

hidden = copy.deepcopy(dm)
hidden.matrix = np.concatenate((sess.run(h[-2], feed_dict={x:dm.matrix}), sess.run(h[-1], feed_dict={x:dm.matrix})), 1)
hidden.matrixname = 'hiddens_of_' + hidden.matrixname
hidden.updateshapeattribute()
hidden.updatesizeattribute()
hidden.updatedtypeattribute()
hidden.columnmeta = {}
hidden.columnname = 'latent_variable'
hidden.columnlabels = np.array(['L'+str(x) for x in range(hidden.shape[1])], dtype='object')

from machinelearning import datasetIO
datasetIO.save_datamatrix('temp_recon_datamatrix.pickle', recon)
datasetIO.save_datamatrix('temp_hidden_datamatrix.pickle', hidden)

partitions = ['train', 'valid', 'test']

dm = {'train':train, 'valid':valid, 'test':test}
recon = {}
for partition in partitions:
    recon[partition] = copy.deepcopy(dm[partition])
train_recon_mat = sess.run(xhat, feed_dict={x:train.matrix})
valid_recon_mat = sess.run(xhat, feed_dict={x:valid.matrix})
test_recon_mat = sess.run(xhat, feed_dict={x:test.matrix})

train_hidden_mat = np.concatenate((sess.run(h[-2], feed_dict={x:train.matrix}), sess.run(h[-1], feed_dict={x:train.matrix})), 1)
valid_hidden_mat = np.concatenate((sess.run(h[-2], feed_dict={x:valid.matrix}), sess.run(h[-1], feed_dict={x:valid.matrix})), 1)
test_hidden_mat = np.concatenate((sess.run(h[-2], feed_dict={x:test.matrix}), sess.run(h[-1], feed_dict={x:test.matrix})), 1)


'''
from scipy.spatial.distance import pdist
metrics = ['euclidean', 'cosine', 'correlation']
for metric in metrics:
    din = pdist(train.matrix, metric)
    dout = pdist(proj2d_train, metric)
    plt.figure()
    rhit = np.random.rand(din.size) < 0.01
    plt.plot(din[rhit], dout[rhit], '.k')
    del din, dout

















# dataset selection
dataset_info = datasetselection.finddatasets(getalllevels=True)
dataset_abbrevs = sorted(list(dataset_info.keys()))

dataset_abbrev = 'gocc'
gene_atb = datasetselection.loaddatamatrix(datasetpath=dataset_info[dataset_abbrev]['path'],
                                           rowname='gene',
                                           columnname='atb',
                                           matrixname='gene_atb_associations',
                                           skiprows=3,
                                           skipcolumns=3,
                                           delimiter='\t',
                                           dtype='float64',
                                           getmetadata=True, # need to fix False case
                                           getmatrix=True)

genes_per_atb = gene_atb.matrix.sum(0)
si = np.argsort(genes_per_atb)[::-1]
geneid_atb = {}
for i in si[200:225]:
    hidxs = (gene_atb.matrix[:,i] != 0).nonzero()[0]
    for hidx in hidxs:
        geneid_atb[gene_atb.rowmeta['GeneID'][hidx]] = gene_atb.columnlabels[i]

train.rowmeta[dataset_abbrev] = np.full(train.shape[0], 'none', dtype='object')
for i, geneid in enumerate(train.rowmeta['GeneID']):
    if geneid in geneid_atb:
        train.rowmeta[dataset_abbrev][i] = geneid_atb[geneid]
valid.rowmeta[dataset_abbrev] = np.full(valid.shape[0], 'none', dtype='object')
for i, geneid in enumerate(valid.rowmeta['GeneID']):
    if geneid in geneid_atb:
        valid.rowmeta[dataset_abbrev][i] = geneid_atb[geneid]

atbs = sorted(list(set(train.rowmeta[dataset_abbrev].tolist()).union(valid.rowmeta[dataset_abbrev].tolist())))
atbs = gene_atb.columnlabels[np.in1d(gene_atb.columnlabels,atbs)]
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/len(atbs))) for i in range(len(atbs))]

fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
for atb, color in zip(atbs, colors):
    if atb == 'none':
        zorder = 0
        alpha = 0.05
        color = 'k'
    else:
        zorder = 1
        alpha = 0.5
    hitt = train.rowmeta[dataset_abbrev] == atb
    hitv = valid.rowmeta[dataset_abbrev] == atb
    ax.plot(np.append(proj2d_train[hitt,0], proj2d_valid[hitv,0], 0), np.append(proj2d_train[hitt,1], proj2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=atb)
#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_coloredby_{2}.png'.format(hiddenlayers, finetuning_run, dataset_abbrev), transparent=True, pad_inches=0, dpi=600)
fg.show()

#fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
#ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
#for atb, color in zip(atbs, colors):
#    if atb == 'none':
#        zorder = 0
#        alpha = 0.05
#        color = 'k'
#    else:
#        zorder = 1
#        alpha = 0.5
#    hitt = train.rowmeta[dataset_abbrev] == atb
#    hitv = valid.rowmeta[dataset_abbrev] == atb
#    ax.plot(np.append(proj2d_train_preactivation[hitt,0], proj2d_valid_preactivation[hitv,0], 0), np.append(proj2d_train_preactivation[hitt,1], proj2d_valid_preactivation[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=atb)
#ax.set_xlim(np.percentile(np.append(proj2d_train_preactivation[:,0], proj2d_valid_preactivation[:,0], 0), 0.05), np.percentile(np.append(proj2d_train_preactivation[:,0], proj2d_valid_preactivation[:,0], 0), 99.95))
#ax.set_ylim(np.percentile(np.append(proj2d_train_preactivation[:,1], proj2d_valid_preactivation[:,1], 0), 0.05), np.percentile(np.append(proj2d_train_preactivation[:,0], proj2d_valid_preactivation[:,0], 0), 99.95))
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
#ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
#ax.set_frame_on(False)
##fg.savefig('results/autoencoder/proj2d_0-1_preactivation_layer{0!s}_finetuning{1!s}_coloredby_{2}.png'.format(hiddenlayers, finetuning_run, dataset_abbrev), transparent=True, pad_inches=0, dpi=600)
#fg.show()










X = euclidean_distances(np.append(proj2d_train, proj2d_valid, 0))

eps_try = np.logspace(-3, -0.5, 6, dtype='float64')
ms_try = np.logspace(1, 2.5, 4, dtype='int64')
ss = np.full((eps_try.size, ms_try.size), -1, dtype='float64')
nc = np.zeros((eps_try.size, ms_try.size), dtype='float64')
for i, eps in enumerate(eps_try):
    for j, ms in enumerate(ms_try):
        clusters = DBSCAN(eps=eps, min_samples=ms, metric='precomputed', algorithm='auto', n_jobs=-1).fit_predict(X)
        nc[i,j] = np.unique(clusters).size
        if nc[i,j] > 1:
            ss[i,j] = silhouette_score(X, clusters, metric='precomputed')
        print('eps: {0:1.3g}, min_samp: {1:1.3g}, num_clust: {2:1.3g}, sil_score: {3:1.3g}'.format(eps, ms, nc[i,j], ss[i,j]))

clusters = DBSCAN(eps=0.01, min_samples=100, metric='precomputed', algorithm='auto', n_jobs=-1).fit_predict(X)
train.rowmeta['Cluster'] = clusters[:train_examples]
valid.rowmeta['Cluster'] = clusters[train_examples:]

atbs = sorted(list(set(train.rowmeta['Cluster'].tolist()).union(valid.rowmeta['Cluster'].tolist())))
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/len(atbs))) for i in range(len(atbs))]

fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
for atb, color in zip(atbs, colors):
    if atb == 'none':
        zorder = 0
        alpha = 0.05
        color = 'k'
    else:
        zorder = 1
        alpha = 0.5
    hitt = train.rowmeta['Cluster'] == atb
    hitv = valid.rowmeta['Cluster'] == atb
    ax.plot(np.append(proj2d_train[hitt,0], proj2d_valid[hitv,0], 0), np.append(proj2d_train[hitt,1], proj2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=atb)
#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
#fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_coloredby_{2}.png'.format(hiddenlayers, finetuning_run, 'Cluster'), transparent=True, pad_inches=0, dpi=600)
fg.show()
'''



centroids = []
with open('centroids.txt', 'rt') as fr:
    for line in fr:
        centroids.append([float(x.strip()) for x in line.split('\t')])
centroids = np.array(centroids, dtype='float64')
np.random.shuffle(centroids)

clusters = KMeans(n_clusters= centroids.shape[0], init=centroids, n_init=1, n_jobs=1).fit_predict(np.append(proj2d_train, proj2d_valid, 0))
train.rowmeta['Cluster'] = clusters[:train_examples]
valid.rowmeta['Cluster'] = clusters[train_examples:]

atbs = sorted(list(set(train.rowmeta['Cluster'].tolist()).union(valid.rowmeta['Cluster'].tolist())))
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/len(atbs))) for i in range(len(atbs))]

fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
ax.set_position([0.15/6.5, 0.15/5.3, 5.0/6.5, 5.0/5.3])
for atb, color in zip(atbs, colors):
    if atb == 'none':
        zorder = 0
        alpha = 0.05
        color = 'k'
    else:
        zorder = 1
        alpha = 0.5
    hitt = train.rowmeta['Cluster'] == atb
    hitv = valid.rowmeta['Cluster'] == atb
    ax.plot(np.append(proj2d_train[hitt,0], proj2d_valid[hitv,0], 0), np.append(proj2d_train[hitt,1], proj2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=atb)
#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, ncol=3, columnspacing=0, markerscale=4, fontsize=8, labelspacing=0.4, handletextpad=0)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/proj2d_0-1_layer{0!s}_finetuning{1!s}_coloredby_{2}.png'.format(hiddenlayers, finetuning_run, 'Cluster'), transparent=True, pad_inches=0, dpi=600)
fg.show()


'''
with open('clusters.txt', 'wt') as fw:
    fw.write('\t'.join(['gene_sym', 'gene_id', 'cluster']) + '\n')
    for gene_sym, gene_id, cluster in zip(train.rowlabels, train.rowmeta['GeneID'], train.rowmeta['Cluster']):
        fw.write('\t'.join([gene_sym, gene_id, str(cluster)]) + '\n')
    for gene_sym, gene_id, cluster in zip(valid.rowlabels, valid.rowmeta['GeneID'], valid.rowmeta['Cluster']):
        fw.write('\t'.join([gene_sym, gene_id, str(cluster)]) + '\n')

with open('clusters.pickle', 'wb') as fw:
    pickle.dump((np.append(train.rowlabels, valid.rowlabels), np.append(train.rowmeta['GeneID'], valid.rowmeta['GeneID']), np.append(train.rowmeta['Cluster'], valid.rowmeta['Cluster'])), fw)
'''


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
