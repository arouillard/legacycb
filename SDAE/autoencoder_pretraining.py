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
#dimensions = [round(input_dimension*x) for x in [1.0, 64.0, 16.0, 4.0, 1.0, 1.0/4.0, 1.0/16.0]]
hiddenlayers = len(dimensions) - 1
noise_probability = 1.0
noise_sigma = 0.9/2.0
noise_distribution = sps.truncnorm(-2.0, 2.0, scale=noise_sigma)
reports_per_log = 20
initialization_sigma = 0.1
initialization_distribution = tf.truncated_normal


# define placeholders, use None to allow feeding different numbers of examples
x = tf.placeholder(tf.float32, [None, input_dimension])
noise = tf.placeholder(tf.float32, [None, input_dimension])
noise_mask = tf.placeholder(tf.float32, [None, input_dimension])


# define variables
global_step, W, bencode, bdecode = create_variables(dimensions, initialization_distribution, initialization_sigma)


# update variables
variables_path = 'results/autoencoder/variables_layer{0!s}.pickle'.format(hiddenlayers-1)
fix_or_init = 'fix'
include_global_step = False
if os.path.exists(variables_path):
    global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode), variables_path, fix_or_init, include_global_step)


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
    init = tf.global_variables_initializer()
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
    plt.semilogx(reporting_steps, train_losses[:,j], ':'+color, label='train,{0:1.5g}'.format(learning_rate))
    plt.semilogx(reporting_steps, valid_losses[:,j], '-'+color, label='valid,{0:1.5g}'.format(learning_rate))
plt.legend(loc='best')
plt.ylabel('loss')
plt.xlabel('steps')
plt.xlim([reporting_steps[0]-1, reporting_steps[-1]+1])
#plt.ylim([0, 1])
plt.show()
# save optimization path
with open('results/autoencoder/learning_rate_tuning_layer{0!s}.pickle'.format(hiddenlayers), 'wb') as fw:
    pickle.dump({'learning_rates':learning_rates, 'colors':colors, 'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses}, fw)
'''

learning_rate = 0.001
epsilon = 0.001
beta1 = 0.9
beta2 = 0.999

#learning_rate = 0.1
epochs = 100
reporting_steps = np.unique(np.logspace(0, np.log10(epochs*batches), reports_per_log*np.log10(epochs*batches)+1, dtype='int32'))
reporting_steps[-1] = epochs*batches
valid_losses = np.zeros(reporting_steps.size, dtype='float32')
train_losses = np.zeros(reporting_steps.size, dtype='float32')
valid_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
train_noisy_losses = np.zeros(reporting_steps.size, dtype='float32')
# define optimizer and training function
#learning_rate_function = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=100000*batches, decay_rate=0.1, staircase=True)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate_function)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, beta1=beta1, beta2=beta2)
train_fn = optimizer.minimize(noisy_loss, global_step=global_step)
saver = tf.train.Saver()
# do some learning and calculate loss on valid data
init = tf.global_variables_initializer()
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
            train_losses[i] = sess.run(loss, feed_dict={x:train.matrix})
            train_noisy_losses[i] = sess.run(noisy_loss, feed_dict={x:train.matrix, noise_mask:(np.random.rand(train.shape[0], input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=train.shape[0]*input_dimension).reshape(train.shape[0], input_dimension)).astype('float32')})
            valid_noisy_losses[i] = sess.run(noisy_loss, feed_dict={x:valid.matrix, noise_mask:(np.random.rand(valid.shape[0], input_dimension) <= noise_probability).astype('float32'), noise:(noise_distribution.rvs(size=valid.shape[0]*input_dimension).reshape(valid.shape[0], input_dimension)).astype('float32')})
            print('step:{0:1.6g}, train loss:{1:1.3g}, valid loss:{2:1.3g}, train noisy loss:{3:1.3g}, valid noisy loss:{4:1.3g}, time:{5:1.6g}'.format(reporting_steps[i], train_losses[i], valid_losses[i], train_noisy_losses[i], valid_noisy_losses[i], time.time() - starttime))
#            if valid_losses[i] >= valid_losses[i-1]:
#                overfitting_score += 1
#            else:
#                overfitting_score = 0
#            if epoch >= 100 and overfitting_score == 3:
#                stopearly = True
#                break
#            if np.mod(i,5) == 0:
#                saver.save(sess, 'results/autoencoder/intermediate_variables_layer{0!s}.ckpt'.format(hiddenlayers))
#                with open('results/autoencoder/intermediate_variables_layer{0!s}.pickle'.format(hiddenlayers), 'wb') as fw:
#                    pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)
            i += 1
        training_step += 1
if stopearly:
    saver.restore(sess, 'results/autoencoder/intermediate_variables_layer{0!s}.ckpt'.format(hiddenlayers))
#    sess.close()
#    variables_path = 'results/autoencoder/intermediate_variables_layer{0!s}.pickle'.format(hiddenlayers)
#    fix_or_init = 'init'
#    include_global_step = True
#    global_step, W, bencode, bdecode = create_variables(dimensions, initialization_distribution, initialization_sigma)
#    global_step, W, bencode, bdecode = update_variables((global_step, W, bencode, bdecode), variables_path, fix_or_init, include_global_step)
#    sess = tf.InteractiveSession()
#    sess.run(init)
    tobediscarded = reporting_steps > sess.run(global_step) - global_step_initial
    train_losses = train_losses[~tobediscarded]
    valid_losses = valid_losses[~tobediscarded]
    train_noisy_losses = train_noisy_losses[~tobediscarded]
    valid_noisy_losses = valid_noisy_losses[~tobediscarded]
    reporting_steps = reporting_steps[~tobediscarded]
# plot loss
fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
ax.semilogx(reporting_steps, train_losses, ':k', linewidth=1, label='train,{0:1.5g}'.format(learning_rate))
ax.semilogx(reporting_steps, valid_losses, '-k', linewidth=1, label='valid,{0:1.5g}'.format(learning_rate))
ax.semilogx(reporting_steps, train_noisy_losses, '--r', linewidth=1, label='train,noisy,{0:1.5g}'.format(learning_rate))
ax.semilogx(reporting_steps, valid_noisy_losses, '-.r', linewidth=1, label='valid,noisy,{0:1.5g}'.format(learning_rate))
ax.legend(loc='best', fontsize=8)
ax.set_ylabel('loss', fontsize=8, fontname='arial')
ax.set_xlabel('steps', fontsize=8, fontname='arial')
ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
ax.set_ylim(0, 5)
ax.tick_params(axis='both', which='major', left='on', right='on', bottom='on', top='off', labelleft='on', labelright='off', labelbottom='on', labeltop='off', labelsize=8)
fg.savefig('results/autoencoder/optimization_path_layer{0!s}.png'.format(hiddenlayers), transparent=True, pad_inches=0, dpi=600)
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
fg.savefig('results/autoencoder/reconstructions_layer{0!s}.png'.format(hiddenlayers), transparent=True, pad_inches=0, dpi=600)
fg.show()
# save optimization path
with open('results/autoencoder/optimization_path_layer{0!s}.pickle'.format(hiddenlayers), 'wb') as fw:
    pickle.dump({'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses, 'valid_noisy_losses':valid_noisy_losses, 'train_noisy_losses':train_noisy_losses, 'learning_rate':learning_rate}, fw)
# save parameters
with open('results/autoencoder/variables_layer{0!s}_finetuning0.pickle'.format(hiddenlayers), 'wb') as fw:
    pickle.dump((sess.run(global_step), sess.run(W), sess.run(bencode), sess.run(bdecode)), fw)


'''
# end tensorflow session
sess.close()
# close figures
for i in range(2):
    plt.close()
# clear everything
%reset -f
'''
