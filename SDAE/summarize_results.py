# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import argparse
import pickle
import json
import os
import numpy as np
import shutil


def main(design_list_or_path):
    
    # resolve design paths
    print('resolving design paths...', flush=True)
    version = ''
    if type(design_list_or_path) == list:
        design_paths = design_list_or_path
    elif '.json' in design_list_or_path:
        design_paths = [design_list_or_path]
    elif '.txt' in design_list_or_path:
        with open(design_list_or_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            design_paths = fr.read().split('\n')
        if '_v' in design_list_or_path:
            version = '_' + design_list_or_path.replace('.txt', '').split('_')[-1]
    else:
        raise ValueError('invalid input to design_list_or_path')
    print('found {0!s} designs...'.format(len(design_paths)), flush=True)
    for design_path in design_paths:
        print('    {0}'.format(design_path), flush=True)
        
    
    # get design fields
    print('getting design fields...', flush=True)
    with open('design_specs_template.json', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        design_fields = sorted(list(json.loads('\n'.join([x.split('#')[0].rstrip() for x in fr.read().split('\n')])).keys()))
    for design_field in design_fields:
        print('    {0}'.format(design_field), flush=True)
        
    
    # specify results fields
    print('specifying results fields...', flush=True)
    results_fields = ['reporting_steps', 'valid_losses', 'train_losses', 'valid_noisy_losses', 'train_noisy_losses']
    for results_field in results_fields:
        print('    {0}'.format(results_field), flush=True)
        
    
    # collect model results
    print('collecting model results...', flush=True)
    if '.json' in design_list_or_path:
        base_path = '/'.join(design_paths[0].split('/')[:-1])
    else:
        base_path = '/'.join(design_paths[0].split('/')[:-2])
    results_path = '{0}/results_summary{1}.txt'.format(base_path, version)
    figure_dirs = {'embedding':base_path + '/embedding',
                   'embedding_preactivation':base_path + '/embedding_preactivation',
                   'optimization_path':base_path + '/optimization_path',
                   'reconstructions':base_path + '/reconstructions',
                   'proj2d':base_path + '/proj2d',
                   'proj2d_preactivation':base_path + '/proj2d_preactivation'}
    for fp in figure_dirs.values():
        if not os.path.exists(fp):
            os.makedirs(fp)
    print('results_path: {0}'.format(results_path), flush=True)
    with open(results_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write('\t'.join(design_fields + ['min_' + f for f in results_fields] + ['selected_' + f for f in results_fields] + ['design_path']) + '\n')
        for design_path in design_paths:
            print('    working on {0}...'.format(design_path), flush=True)
            print('    getting design specs...', flush=True)
            with open(design_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
                design = json.load(fr)
            if 'apply_activation_to_embedding' not in design: # for legacy code
                design['apply_activation_to_embedding'] = True
            if 'use_batchnorm' not in design: # for legacy code
                design['use_batchnorm'] = False
            if 'skip_layerwise_training' not in design: # for legacy code
                design['skip_layerwise_training'] = False
            design_values = [design[f] if type(design[f]) == str else '{0:1.6g}'.format(design[f]) for f in design_fields]
            print('    getting reconstruction errors...', flush=True)
            optimization_path = '{0}/optimization_path_layer{1}_finetuning1.pickle'.format(design['output_path'], design['hidden_layers'])
            if os.path.exists(optimization_path):
                with open(optimization_path, 'rb') as fr:
                    optimization_results = pickle.load(fr)
                hidx = np.argmin(optimization_results['valid_losses'])
                results_values = ['{0:1.6g}'.format(optimization_results[f][hidx]) for f in results_fields]
                log_path = '{0}/log.txt'.format(design['output_path'])
                selected_step = optimization_results['reporting_steps'][-1]
                with open(log_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
                    for line in fr:
                        if line[:14] == 'selected step:':
                            selected_step = int(line[14:].replace('...','').strip())
                hidx = np.argmin(np.abs(optimization_results['reporting_steps'] - selected_step))
                results_values += ['{0:1.6g}'.format(optimization_results[f][hidx]) for f in results_fields]
            else:
                results_values = ['NAN' for f in results_fields]
                results_values += results_values
            fw.write('\t'.join(design_values + results_values + [design_path]) + '\n')
            for figure_prefix, figure_dir in figure_dirs.items():
#                target_label_condensed = '_'.join(design['output_path'].split('/')[-1].split('_')[:12] + [design['activation_function']])
                target_label_condensed = design['output_path'].split('/')[-1].replace('_', '')
                target_label_condensed = target_label_condensed.replace('sigmoid', 'S').replace('tanh', 'T').replace('relu', 'R').replace('truncnorm', 'TN').replace('uniform', 'U').replace('bernoulli', 'B').replace('replace', 'R').replace('add', 'A').replace('flip', 'F').replace('True', 'T').replace('False', 'F')
                source_path = '{0}/{1}_layer{2}_finetuning1.png'.format(design['output_path'], figure_prefix, design['hidden_layers'])
                target_path = '{0}/{1}.png'.format(figure_dir, target_label_condensed)
                if os.path.exists(source_path):
                    shutil.copyfile(source_path, target_path)
                    
                


    print('done summarize_results.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch training of one or more models.')
    parser.add_argument('design_list_or_path', help='path to .json file for a single design or path to .txt file containing paths for a batch of designs', type=str)
    args = parser.parse_args()
    main(args.design_list_or_path)

