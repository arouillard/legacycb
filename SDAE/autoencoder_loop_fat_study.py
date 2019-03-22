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
import autoencoder_function_beta_2 as autoencoder_function


# define configuration
print('defining configuration...')
current_hidden_layer = 0
current_finetuning_run = 0 # 0 means pretraining
previous_hidden_layer = 0
previous_finetuning_run = 0
pretraining_epochs = 100 # 10000 # 100
finetuning_epochs = 100
last_layer_epochs = 100000
use_finetuning = True
min_dimension = 16  # 3528, 1764, 882, 441
first_hidden_layer_heuristic = 1/10 # expand, iso, or contract
extra_layers = 3
layer_scaling_factor = round(10*(first_hidden_layer_heuristic*17642/min_dimension)**(1/extra_layers))/10
#first_hidden_layer_heuristic = 'contract' # expand, iso, or contract
#hidden_layers = 1
#layer_scaling_factor = round(10*(17642/min_dimension)**(1/hidden_layers))/10
noise_probability = 1.0
noise_sigma = 5.0
noise_shape = sps.truncnorm # sps.truncnorm or sps.uniform
noise_operation = 'add' # add or replace
initialization_sigma = 0.01
initialization_distribution = tf.truncated_normal
learning_rate = 0.001 # default 0.001
epsilon = 0.001 # default 10**-8
beta1 = 0.9 # default 0.9
beta2 = 0.999 # default 0.999
batch_fraction = 0.1
firstcheckpoint = int(1/batch_fraction) # 1, int(1/batch_fraction), or int(pretraining_epochs/batch_fraction)
maxstepspercheckpoint = int(1e4) # int(1e10)
startsavingstep = int(1e4) # int(1e10)
include_global_step = False
overfitting_score_max = 5
activation_function_name, activation_function = ('tanh', tf.tanh) # ('relu', tf.nn.relu), tf.nn.elu, tf.sigmoid, or ('tanh', tf.tanh)
apply_activation_to_output = False
processor = 'gpu' # cpu or gpu
study_name = 'gtex_plus_asthma_remission_study' # 'gtex_plus_ilc2_ibet_study'
input_path = 'data/prepared_data/{0}/fat'.format(study_name)
output_path = 'results/autoencoder/fat/{0}/lsf{1!s}_md{2!s}_{3!s}_np{4!s}_ns{5!s}_is{6!s}_lr{7!s}_eps{8!s}_bf{9!s}_pte{10!s}_fte{11!s}_{12}'\
              .format(study_name,
                      layer_scaling_factor,
                      min_dimension,
                      first_hidden_layer_heuristic,
                      noise_probability,
                      noise_sigma,
                      initialization_sigma,
                      learning_rate,
                      epsilon,
                      batch_fraction,
                      pretraining_epochs,
                      finetuning_epochs,
                      activation_function_name)


# confirm paths
print('confirm paths...', flush=True)
print('input path: {0}'.format(input_path), flush=True)
print('output path: {0}'.format(output_path), flush=True)
time.sleep(10)


# confirm dimensions
print('confirm dimensions...', flush=True)
with open('{0}/valid.pickle'.format(input_path), 'rb') as fr:
    valid = pickle.load(fr)
input_dimension = valid.shape[1]
all_dimensions = autoencoder_function.get_all_dimensions(input_dimension, layer_scaling_factor, min_dimension, first_hidden_layer_heuristic)
print('all_dimensions:', all_dimensions, flush=True)
del valid, input_dimension
time.sleep(10)


# confirm reporting steps
print('confirm reporting steps...', flush=True)
reporting_steps = autoencoder_function.create_reporting_steps(int(pretraining_epochs/batch_fraction),
                                                              firstcheckpoint, maxstepspercheckpoint)
print('reporting_steps:', reporting_steps, flush=True)
time.sleep(10)


# assemble inputs
print('assembling inputs...', flush=True)
non_current_inputs = [input_path,
                      output_path,
                      layer_scaling_factor,
                      min_dimension,
                      first_hidden_layer_heuristic,
                      noise_probability,
                      noise_sigma,
                      noise_shape,
                      noise_operation,
                      initialization_sigma,
                      initialization_distribution,
                      learning_rate,
                      epsilon,
                      beta1,
                      beta2,
                      batch_fraction,
                      firstcheckpoint,
                      maxstepspercheckpoint,
                      startsavingstep,
                      use_finetuning,
                      include_global_step,
                      overfitting_score_max,
                      activation_function,
                      apply_activation_to_output,
                      processor]


# assemble layer training schedule
print('assembling layer training schedule...', flush=True)
if use_finetuning:
    hidden_layers = list(range(1,len(all_dimensions)))
    finetuning_runs = []
    layer_epochs = []
    for i in range(len(hidden_layers)):
        finetuning_runs += [0, 1]
        layer_epochs += [pretraining_epochs, finetuning_epochs]
    hidden_layers = sorted(hidden_layers + hidden_layers)
    layer_epochs[-1] = last_layer_epochs
else:
    hidden_layers = list(range(1,len(all_dimensions)))
    finetuning_runs = [0 for i in range(len(hidden_layers))]
    layer_epochs = [pretraining_epochs for i in range(len(hidden_layers))]
    layer_epochs[-1] = last_layer_epochs


# build model
print('building model...\n\n', flush=True)
#previous_hidden_layer = hidden_layers[-2]
#previous_finetuning_run = finetuning_runs[-2]
#for current_hidden_layer, current_finetuning_run, current_epochs in zip(hidden_layers[-1:], finetuning_runs[-1:], layer_epochs[-1:]):
#for current_hidden_layer, current_finetuning_run, current_epochs in [(hidden_layers[-1], 1.1, 100000)]:
#for current_hidden_layer, current_finetuning_run, current_epochs in zip(hidden_layers[:1], finetuning_runs[:1], layer_epochs[:1]):
for current_hidden_layer, current_finetuning_run, current_epochs in zip(hidden_layers, finetuning_runs, layer_epochs):

    print('working on hidden layer {0!s}, finetuning run {1!s}, epochs {2!s}...'\
          .format(current_hidden_layer, current_finetuning_run, current_epochs), flush=True)
    all_inputs = [current_hidden_layer, current_finetuning_run, previous_hidden_layer, previous_finetuning_run, current_epochs]\
                 + non_current_inputs
    previous_hidden_layer, previous_finetuning_run, previous_epochs = autoencoder_function.main(*all_inputs)
    
    print('finished hidden layer {0!s}, finetuning run {1!s}, epochs {2!s}.\n\n'\
          .format(previous_hidden_layer, previous_finetuning_run, previous_epochs), flush=True)
    
print('done.', flush=True)
