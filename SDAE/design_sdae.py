# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import os
import json
import copy
import argparse
from itertools import product
import finish_design


def main(design_specs_path, hyperparameter_search_type=None):
    
    # load design specifications
    print('loading design specifications...', flush=True)
    print('design_specs_path: {0}'.format(design_specs_path), flush=True)
    with open(design_specs_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        design_specs = json.loads('\n'.join([x.split('#')[0].rstrip() for x in fr.read().split('\n')]))
    
    
    # initialize set of design paths
    print('initializing set of design paths...', flush=True)
    design_paths = set()
    
    
    # create base design and set up hyperparameter searches
    print('creating base design and setting up hyperparameter searches...', flush=True)
    hyperparameter_values = {}
    base_design = {}
    for parameter, values in design_specs.items():
        if type(values) == list:
            hyperparameter_values[parameter] = values
            base_design[parameter] = values[0]
        else:
            base_design[parameter] = values
    
    
    # finish base design
    print('running finish_design on base_design...', flush=True)
    base_design_path = finish_design.main(base_design)
    
    
    # append base design path to list
    print('appending base design path to list...', flush=True)
    design_paths.add(base_design_path)


    # create design variations
    if hyperparameter_search_type == 'line':
        print('creating design variations using line search...', flush=True)
        # iterate over hyperparameters
        for hyperparameter, values in hyperparameter_values.items():
            print('    working on hyperparameter {0}...'.format(hyperparameter), flush=True)
            d = copy.deepcopy(base_design)
            
            # iterate over hyperparameter values
            for value in values:
                print('        updating {0} to {1!s}...'.format(hyperparameter, value), flush=True)
                d[hyperparameter] = value
                
                # finish design
                print('        running finish_design...', flush=True)
                design_path = finish_design.main(d)
                
                # append new design path to list
                print('        appending new design path to list...', flush=True)
                design_paths.add(design_path)
    
    elif hyperparameter_search_type == 'grid':
        print('creating design variations using grid search...', flush=True)
        # iterate over hyperparameter combinations
        hyperparameters = list(hyperparameter_values.keys())
        for combination_index, combination_values in enumerate(product(*[hyperparameter_values[hp] for hp in hyperparameters])):
            print('    working on hyperparameter combination {0!s}...'.format(combination_index), flush=True)
            d = copy.deepcopy(base_design)
            
            # update hyperparameters
            for hyperparameter, value in zip(hyperparameters, combination_values):
                print('        updating {0} to {1!s}...'.format(hyperparameter, value), flush=True)
                d[hyperparameter] = value
            
            # finish design
            print('    running finish_design...', flush=True)
            design_path = finish_design.main(d)
            
            # append new design path to list
            print('    appending new design path to list...', flush=True)
            design_paths.add(design_path)
            
    else:
        print('skipping hyperparameter search...', flush=True)
    
    
    # save list of design paths
    print('saving list of design paths...', flush=True)
#    design_paths_path = '/'.join(design_specs_path.split('/')[:-1] + ['design_paths.txt'])
    design_paths_path = design_specs_path.replace('specs', 'paths').replace('json', 'txt')
    print('design_paths_path: {0}'.format(design_paths_path), flush=True)
    if os.path.exists(design_paths_path):
        print('design_paths_path already exists. updating list...', flush=True)
        with open(design_paths_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            design_paths.update(fr.read().split('\n'))
    design_paths = sorted(list(design_paths))
    with open(design_paths_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write('\n'.join(design_paths))
    
    
    print('done design_sdae.', flush=True)
    
    return design_paths, design_paths_path
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create SDAE designs according to design specifications.')
    parser.add_argument('design_specs_path', help='path to .json file with design specifications', type=str)
    parser.add_argument('--hyperparameter_search_type', help='whether to perform line or grid search on hyperparameters with multiple values specified (default None)', type=str, choices=['line', 'grid'])
    args = parser.parse_args()    
    main(args.design_specs_path, args.hyperparameter_search_type)

