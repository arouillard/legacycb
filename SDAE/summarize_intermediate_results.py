# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import os
import argparse
import pickle
import json
import numpy as np
import sdae_apply_functions
import datasetIO


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
    results_fields = ['reporting_steps', 'test_losses', 'valid_losses', 'train_losses']
    for results_field in results_fields:
        print('    {0}'.format(results_field), flush=True)
        
    
    # collect model results
    print('collecting model results...', flush=True)
    if '.json' in design_list_or_path:
        base_path = '/'.join(design_paths[0].split('/')[:-1])
    else:
        base_path = '/'.join(design_paths[0].split('/')[:-2])
    results_path = '{0}/intermediate_results_summary{1}.txt'.format(base_path, version)
    print('results_path: {0}'.format(results_path), flush=True)
    with open(results_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write('\t'.join(design_fields + results_fields + ['design_path']) + '\n')
        for didx, design_path in enumerate(design_paths):
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
            phase = design['training_schedule'][-1]
            design['current_hidden_layer'] = phase['hidden_layer']
            design['current_finetuning_run'] = phase['finetuning_run']
            design['current_epochs'] = phase['epochs']
            design_values = [design[f] if type(design[f]) == str else '{0:1.6g}'.format(design[f]) for f in design_fields]
            
            # load data
            if didx == 0:
                print('loading data...', flush=True)
                partitions = ['train', 'valid', 'test']
                dataset = {}
                for partition in partitions:
                    dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(design['input_path'], partition))
                    
            # finish configuration
            print('finishing configuration...', flush=True)
            
            # specify activation function
            if design['activation_function'] == 'tanh':
                activation_function = {'np':sdae_apply_functions.tanh}
            elif design['activation_function'] == 'relu':
                activation_function = {'np':sdae_apply_functions.relu}
            elif design['activation_function'] == 'elu':
                activation_function = {'np':sdae_apply_functions.elu}
            elif design['activation_function'] == 'sigmoid':
                activation_function = {'np':sdae_apply_functions.sigmoid}
        
            # initialize model architecture (number of layers and dimension of each layer)
            design['current_dimensions'] = design['all_dimensions'][:design['current_hidden_layer']+1] # dimensions of model up to current depth
            
            # specify embedding function for current training phase
            # we want the option of skipping the embedding activation function to apply only to the full model
            if not design['apply_activation_to_embedding'] and design['current_dimensions'] == design['all_dimensions']:
                design['current_apply_activation_to_embedding'] = False
            else:
                design['current_apply_activation_to_embedding'] = True
            print('current_apply_activation_to_embedding: {0!s}'.format(design['current_apply_activation_to_embedding']), flush=True)
            
            # get reconstruction errors
            print('    getting reconstruction errors...', flush=True)
            optimization_path = '{0}/optimization_path_layer{1}_finetuning{2}.pickle'.format(design['output_path'], design['current_hidden_layer'], design['current_finetuning_run'])
            if os.path.exists(optimization_path):
                with open(optimization_path, 'rb') as fr:
                    optimization_results = pickle.load(fr)
                for step, valid_loss, train_loss in zip(optimization_results['reporting_steps'], optimization_results['valid_losses'], optimization_results['train_losses']):
                    intermediate_variables_path = '{0}/intermediate_variables_layer{1}_finetuning{2}_step{3!s}.pickle'.format(design['output_path'], design['current_hidden_layer'], design['current_finetuning_run'], step)
                    if os.path.exists(intermediate_variables_path):
                        # load model variables
                        print('loading model variables...', flush=True)
                        with open(intermediate_variables_path, 'rb') as fr:
                            W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
                        if design['use_batchnorm']:
                            with open('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2}_step{3!s}.pickle'.format(design['output_path'], design['current_hidden_layer'], design['current_finetuning_run'], step), 'rb') as fr:
                                batchnorm_variables = pickle.load(fr) # gammas, betas, moving_means, moving_variances
                            batchnorm_encode_variables, batchnorm_decode_variables = sdae_apply_functions.align_batchnorm_variables(batchnorm_variables, design['current_apply_activation_to_embedding'], design['apply_activation_to_output'])
                        # compute embedding and reconstruction
                        print('computing embedding and reconstruction...', flush=True)        
                        recon = {}
                        embed = {}
                        error = {}
                        embed_preactivation = {}
                        for partition in partitions:
                            if design['use_batchnorm']:
                                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], design['current_apply_activation_to_embedding'], design['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
                            else:
                                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], design['current_apply_activation_to_embedding'], design['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True)
                                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
                            print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)                        
                        test_loss = error['test']
                        valid_loss = error['valid']
                        train_loss = error['train']
                    else:
                        test_loss = np.nan
                    results_values = ['{0:1.6g}'.format(x) for x in [step, test_loss, valid_loss, train_loss]]
                    fw.write('\t'.join(design_values + results_values + [design_path]) + '\n')
            else:
                intermediate_steps = [int(float(x.split('_')[-1].replace('step', '').replace('.pickle', ''))) for x in os.listdir(design['output_path']) if 'intermediate_variables_layer{0}_finetuning{1}_step'.format(design['current_hidden_layer'], design['current_finetuning_run']) in x]
                for step in intermediate_steps:
                    intermediate_variables_path = '{0}/intermediate_variables_layer{1}_finetuning{2}_step{3!s}.pickle'.format(design['output_path'], design['current_hidden_layer'], design['current_finetuning_run'], step)
                    if os.path.exists(intermediate_variables_path):
                        # load model variables
                        print('loading model variables...', flush=True)
                        with open(intermediate_variables_path, 'rb') as fr:
                            W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
                        if design['use_batchnorm']:
                            with open('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2}_step{3!s}.pickle'.format(design['output_path'], design['current_hidden_layer'], design['current_finetuning_run'], step), 'rb') as fr:
                                batchnorm_variables = pickle.load(fr) # gammas, betas, moving_means, moving_variances
                            batchnorm_encode_variables, batchnorm_decode_variables = sdae_apply_functions.align_batchnorm_variables(batchnorm_variables, design['current_apply_activation_to_embedding'], design['apply_activation_to_output'])
                        # compute embedding and reconstruction
                        print('computing embedding and reconstruction...', flush=True)        
                        recon = {}
                        embed = {}
                        error = {}
                        embed_preactivation = {}
                        for partition in partitions:
                            if design['use_batchnorm']:
                                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], design['current_apply_activation_to_embedding'], design['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
                            else:
                                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], design['current_apply_activation_to_embedding'], design['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True)
                                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
                            print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)                        
                        test_loss = error['test']
                        valid_loss = error['valid']
                        train_loss = error['train']
                    else:
                        test_loss = np.nan
                    results_values = ['{0:1.6g}'.format(x) for x in [step, test_loss, valid_loss, train_loss]]
                    fw.write('\t'.join(design_values + results_values + [design_path]) + '\n')

                    
    print('done summarize_intermediate_results.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize intermediate results of one or more models.')
    parser.add_argument('design_list_or_path', help='path to .json file for a single design or path to .txt file containing paths for a batch of designs', type=str)
    args = parser.parse_args()
    main(args.design_list_or_path)

