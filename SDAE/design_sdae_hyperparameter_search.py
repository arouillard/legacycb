# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import sys
import json
import copy
from itertools import product
import finish_sdae_design


def main(base_design_path, search_type='line'):
    
    # setup hyperparameter searches
    print('setting up hyperparameter searches...', flush=True)
    hyperparameter_values = {}
    #hyperparameter_values['pretraining_epochs'] = 100
    #hyperparameter_values['finetuning_epochs'] = 100
    #hyperparameter_values['last_layer_epochs'] = 100000
    #hyperparameter_values['use_finetuning'] = True
    #hyperparameter_values['min_dimension'] = 2
    hyperparameter_values['hidden_layers'] = [4, 5, 6]
    #hyperparameter_values['first_hidden_layer_scaling_factor'] = 'auto' # 'auto' or a scaling factor
    #hyperparameter_values['noise_probability'] = 1.0
    hyperparameter_values['noise_sigma'] = [3.0, 5.0, 7.0]
    #hyperparameter_values['noise_distribution'] = 'truncnorm' # 'truncnorm' or 'uniform' sps.truncnorm sps.uniform
    #hyperparameter_values['noise_operation'] = 'add' # 'add' or 'replace'
    #hyperparameter_values['initialization_sigma'] = [0.001, 0.01, 0.1]
    #hyperparameter_values['initialization_distribution'] = 'truncnorm' # 'truncnorm' tf.truncated_normal
    #hyperparameter_values['learning_rate'] = 0.001
    #hyperparameter_values['epsilon'] = 0.001
    #hyperparameter_values['beta1'] = 0.9
    #hyperparameter_values['beta2'] = 0.999
    #hyperparameter_values['batch_fraction'] = 0.1
    #hyperparameter_values['firstcheckpoint'] = int(1/hyperparameter_values['batch_fraction']) # 1, int(1/batch_fraction), or int(pretraining_epochs/batch_fraction)
    #hyperparameter_values['maxstepspercheckpoint'] = int(1e5)
    #hyperparameter_values['startsavingstep'] = int(1e4)
    #hyperparameter_values['include_global_step'] = False
    #hyperparameter_values['overfitting_score_max'] = 5
    hyperparameter_values['activation_function'] = ['relu', 'tanh'] # 'relu', 'tanh', 'elu', or 'sigmoid' 
    #hyperparameter_values['apply_activation_to_output'] = False
    #hyperparameter_values['processor'] = 'gpu' # 'cpu' or 'gpu'
    #hyperparameter_values['gpu_memory_fraction'] = 0.2
    #hyperparameter_values['gpu_id'] = '0'
    #hyperparameter_values['study_name'] = 'GTEXv6plus'
    #hyperparameter_values['orientation'] = 'fat' # 'skinny' or 'fat'
    
    
    # initialize set of design paths
    print('initializing set of design paths...', flush=True)
    design_paths = set()
    
    
    # load base design
    print('loading design...', flush=True)
    print('base_design_path: {0}'.format(base_design_path), flush=True)
    with open(base_design_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        base_design = json.load(fr)
    
    
    # append base design path to list
    print('appending base design path to list...', flush=True)
    design_paths.add(base_design_path)


    # create design variations
    print('creating design variations...', flush=True)
    
    if search_type == 'line':
        # iterate over hyperparameters
        for hyperparameter, values in hyperparameter_values.items():
            print('    working on hyperparameter {0}...'.format(hyperparameter), flush=True)
            d = copy.deepcopy(base_design)
            
            # iterate over hyperparameter values
            for value in values:
                print('        updating {0} to {1!s}...'.format(hyperparameter, value), flush=True)
                d[hyperparameter] = value
                
                # finish design
                print('        running finish_sdae_design...', flush=True)
                design_path = finish_sdae_design.main(d)
                
                # append new design path to list
                print('        appending new design path to list...', flush=True)
                design_paths.add(design_path)
    
    elif search_type == 'grid':
        # iterate over hyperparameter combinations
        hyperparameters = list(hyperparameter_values.keys())
        for combination_index, combination_values in enumerate(product(*[hyperparameter_values[hp] for hp in hyperparameters])):
            print('    working on hyperparameter combination {0!s}...'.format(combination_index), flush=True)
            d = copy.deepcopy(base_design)
            
            # update over hyperparameters
            for hyperparameter, value in zip(hyperparameters, combination_values):
                print('        updating {0} to {1!s}...'.format(hyperparameter, value), flush=True)
                d[hyperparameter] = value
            
            # finish design
            print('    running finish_sdae_design...', flush=True)
            design_path = finish_sdae_design.main(d)
            
            # append new design path to list
            print('    appending new design path to list...', flush=True)
            design_paths.add(design_path)
            
    else:
        raise ValueError('invalid search_type, specify "line" or "grid"')
    
    
    # save list of design paths
    print('saving list of design paths...', flush=True)
    design_paths = list(design_paths)
    design_paths_path = base_design_path.replace('design.json', 'hyperparameter_search_design_paths.txt')
    print('design_paths_path: {0}'.format(design_paths_path), flush=True)
    with open(design_paths_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write('\n'.join(design_paths))
    
    
    print('done setup_sdae_hyperparameter_search.', flush=True)
    
    return design_paths, design_paths_path
    
    
if __name__ == '__main__':
    main(sys.argv[1])

