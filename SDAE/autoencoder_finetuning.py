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


# update variables
variables_path = 'results/autoencoder/variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run-1)
fix_or_init = 'init'
include_global_step = True # False for first finetuning
if os.path.exists(variables_path):
    global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode), variables_path, fix_or_init, include_global_step)


# define variables initializer
init = tf.global_variables_initializer()


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

'''
# tune learning rate
learning_rates = np.logspace(-2, 0, 5, dtype='float32')
colors = np.array(['r', 'g', 'b', 'c', 'm', 'y', 'k'], dtype='object')
epochs = 10
reporting_steps = np.unique(np.logspace(0, np.log10(epochs*batches), reports_per_log*np.log10(epochs*batches)+1, dtype='int32'))
reporting_steps[-1] = epochs*batches
valid_losses = np.zeros((reporting_steps.size, learning_rates.size), dtype='float32')
train_losses = np.zeros((reporting_steps.size, learning_rates.size), dtype='float32')
for j, learning_rate in enumerate(learning_rates):
    # define optimizer and training function
    learning_rate_function = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=100000*batches, decay_rate=0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate_function)
    train_fn = optimizer.minimize(noisy_loss, global_step=global_step)
    # do some learning and calculate loss on valid data
    sess.run(init)
    training_step = 1
    i = 0
    for epoch in range(epochs):
        np.random.shuffle(batch_ids)
        for batch in range(batches):
            selected = batch_ids == batch
            sess.run(train_fn, feed_dict={x:train.matrix[selected,:], noise_mask:(np.random.rand(selected.sum(), input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=selected.sum()*input_dimension).reshape(selected.sum(), input_dimension)).astype('float32')})
            if training_step == reporting_steps[i]:
                valid_losses[i,j] = sess.run(loss, feed_dict={x:valid.matrix})
                train_losses[i,j] = sess.run(loss, feed_dict={x:train.matrix[selected,:]})
                i += 1
            training_step += 1
    isbadval = np.logical_or(np.isinf(train_losses[:,j]), np.isnan(train_losses[:,j]))
    train_losses[isbadval,j] = train_losses[~isbadval,j].max()
    isbadval = np.logical_or(np.isinf(valid_losses[:,j]), np.isnan(valid_losses[:,j]))
    valid_losses[isbadval,j] = valid_losses[~isbadval,j].max()
# plot loss
plt.figure()
for j, (learning_rate, color) in enumerate(zip(learning_rates, colors)):
    plt.loglog(reporting_steps, train_losses[:,j], ':'+color, label='train,{0:1.5g}'.format(learning_rate))
    plt.loglog(reporting_steps, valid_losses[:,j], '-'+color, label='valid,{0:1.5g}'.format(learning_rate))
plt.legend(loc='best')
plt.ylabel('loss')
plt.xlabel('steps')
plt.xlim([reporting_steps[0]-1, reporting_steps[-1]+1])
plt.ylim([0.00003, 300])
plt.show()
# save optimization path
with open('results/autoencoder/learning_rate_tuning_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run), 'wb') as fw:
    pickle.dump({'learning_rates':learning_rates, 'colors':colors, 'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses}, fw)
'''

learning_rate = 0.01 # 0.03 # 0.1 for first finetuning
epochs = 30000 # 60000 # 10000
reporting_steps = np.unique(np.logspace(0, np.log10(epochs*batches), reports_per_log*np.log10(epochs*batches)+1, dtype='int32'))
reporting_steps[-1] = epochs*batches
valid_losses = np.zeros(reporting_steps.size, dtype='float32')
train_losses = np.zeros(reporting_steps.size, dtype='float32')
# define optimizer and training function
learning_rate_function = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=100000*batches, decay_rate=0.1, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate_function)
train_fn = optimizer.minimize(noisy_loss, global_step=global_step)
saver = tf.train.Saver()
# do some learning and calculate loss on valid data
sess.run(init)
global_step_initial = sess.run(global_step)
training_step = 1
i = 0
overfitting_score = 0
stopearly = False
starttime = time.time()
for epoch in range(epochs):
    if stopearly:
        break
    np.random.shuffle(batch_ids)
    for batch in range(batches):
        selected = batch_ids == batch
        sess.run(train_fn, feed_dict={x:train.matrix[selected,:], noise_mask:(np.random.rand(selected.sum(), input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=selected.sum()*input_dimension).reshape(selected.sum(), input_dimension)).astype('float32')})
        if training_step == reporting_steps[i]:
            valid_losses[i] = sess.run(loss, feed_dict={x:valid.matrix})
            train_losses[i] = sess.run(loss, feed_dict={x:train.matrix[selected,:]})
            print('step:{0:1.5g}, training loss:{1:1.5g}, validation loss:{2:1.5g}, total time:{3:1.5g}'.format(reporting_steps[i], train_losses[i], valid_losses[i], time.time() - starttime))
            if valid_losses[i] >= valid_losses[i-1]:
                overfitting_score += 1
            else:
                overfitting_score = 0
            if epoch >= 100 and overfitting_score == 5:
                stopearly = True
                break
            if np.mod(i,9) == 0:
                saver.save(sess, 'results/autoencoder/intermediate_variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run))
#                with open('results/autoencoder/intermediate_variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run), 'wb') as fw:
#                    pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)
            i += 1
        training_step += 1
if stopearly:
    saver.restore(sess, 'results/autoencoder/intermediate_variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run))
#    variables_path = 'results/autoencoder/intermediate_variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run)
#    fix_or_init = 'init'
#    include_global_step = True
#    global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode), variables_path, fix_or_init, include_global_step)
#    sess.run(init)
    tobediscarded = reporting_steps > sess.run(global_step) - global_step_initial
    train_losses = train_losses[~tobediscarded]
    valid_losses = valid_losses[~tobediscarded]
    reporting_steps = reporting_steps[~tobediscarded]
# plot loss
fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
ax.loglog(reporting_steps, train_losses, ':k', linewidth=1, label='train,{0:1.5g}'.format(learning_rate))
ax.loglog(reporting_steps, valid_losses, '-k', linewidth=1, label='valid,{0:1.5g}'.format(learning_rate))
ax.legend(loc='best', fontsize=8)
ax.set_ylabel('loss', fontsize=8, fontname='arial')
ax.set_xlabel('steps', fontsize=8, fontname='arial')
ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
ax.set_ylim(0.00003, 3)
ax.tick_params(axis='both', which='major', left='on', right='off', bottom='on', top='off', labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
fg.savefig('results/autoencoder/optimization_path_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
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
fg.savefig('results/autoencoder/reconstructions_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
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
fg.savefig('results/autoencoder/proj2d_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()
fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
ax.plot(proj2d_train_preactivation[:,0], proj2d_train_preactivation[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.plot(proj2d_valid_preactivation[:,0], proj2d_valid_preactivation[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/autoencoder/proj2d_preactivation_layer{0!s}_finetuning{1!s}.png'.format(hiddenlayers, finetuning_run), transparent=True, pad_inches=0, dpi=600)
fg.show()
# save optimization path
with open('results/autoencoder/optimization_path_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run), 'wb') as fw:
    pickle.dump({'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses, 'learning_rate':learning_rate}, fw)
# save parameters
with open('results/autoencoder/variables_layer{0!s}_finetuning{1!s}.pickle'.format(hiddenlayers, finetuning_run), 'wb') as fw:
    pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)


'''
# end tensorflow session
sess.close()
# close figures
for i in range(5):
    plt.close()
# clear everything
%reset -f
'''
