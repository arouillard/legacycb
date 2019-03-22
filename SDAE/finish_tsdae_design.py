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
import pickle
import os
import tsdae_design_functions


def main(design_dict_or_path):
    
    
    # load design
    if type(design_dict_or_path) == str:
        print('loading design...', flush=True)
        print('base_design_path: {0}'.format(design_dict_or_path), flush=True)
        with open(design_dict_or_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            d = json.load(fr)
    elif type(design_dict_or_path) == dict:
        d = design_dict_or_path
    else:
        raise ValueError('input to finish_design must be dict or string')

    
    # create paths
    print('creating input and output paths...', flush=True)
    d['input_path'] = 'data/prepared_data/{0}/{1}'.format(d['study_name'], d['orientation'])
    if 'structured_noise_probability' in d:
        d['output_path'] = 'results/autoencoder/{0}/{1}/hl{2!s}_md{3!s}_fhlsf{4!s}_np{5!s}_ns{6!s}_nd{7!s}_no{8}_is{9!s}_id{10}_lr{11!s}_eps{12!s}_bf{13!s}_pte{14!s}_fte{15!s}_uft{16!s}_slt{17!s}_ubn{18!s}_aao{19!s}_aae{20!s}_{21}'\
                      .format(d['study_name'],
                              d['orientation'],
                              d['hidden_layers'],
                              d['min_dimension'],
                              d['first_hidden_layer_scaling_factor'],
                              d['noise_probability'],
                              d['structured_noise_probability'],
                              d['bernoulli_weight'],
                              d['noise_operation'],
                              d['initialization_sigma'],
                              d['initialization_distribution'],
                              d['learning_rate'],
                              d['epsilon'],
                              d['batch_fraction'],
                              d['pretraining_epochs'],
                              d['finetuning_epochs'],
                              d['use_finetuning'],
                              d['skip_layerwise_training'],
                              d['use_batchnorm'],
                              d['apply_activation_to_output'],
                              d['apply_activation_to_embedding'],
                              d['activation_function'])
    else:
        d['output_path'] = 'results/autoencoder/{0}/{1}/hl{2!s}_md{3!s}_fhlsf{4!s}_np{5!s}_ns{6!s}_nd{7}_no{8}_is{9!s}_id{10}_lr{11!s}_eps{12!s}_bf{13!s}_pte{14!s}_fte{15!s}_uft{16!s}_slt{17!s}_ubn{18!s}_aao{19!s}_aae{20!s}_{21}'\
                      .format(d['study_name'],
                              d['orientation'],
                              d['hidden_layers'],
                              d['min_dimension'],
                              d['first_hidden_layer_scaling_factor'],
                              d['noise_probability'],
                              d['noise_sigma'],
                              d['noise_distribution'],
                              d['noise_operation'],
                              d['initialization_sigma'],
                              d['initialization_distribution'],
                              d['learning_rate'],
                              d['epsilon'],
                              d['batch_fraction'],
                              d['pretraining_epochs'],
                              d['finetuning_epochs'],
                              d['use_finetuning'],
                              d['skip_layerwise_training'],
                              d['use_batchnorm'],
                              d['apply_activation_to_output'],
                              d['apply_activation_to_embedding'],
                              d['activation_function'])


    # confirm paths
    print('confirm paths...', flush=True)
    print('input path: {0}'.format(d['input_path']), flush=True)
    print('output path: {0}'.format(d['output_path']), flush=True)


    # create output directory
    print('creating output directory...', flush=True)
    if not os.path.exists(d['output_path']):
        os.makedirs(d['output_path'])


    # confirm dimensions
    print('confirm dimensions...', flush=True)
    with open('{0}/valid.pickle'.format(d['input_path']), 'rb') as fr:
        d['input_dimension'] = pickle.load(fr).shape[1]
    d['all_dimensions'] = tsdae_design_functions.get_layer_dimensions(d['input_dimension'], d['min_dimension'], d['hidden_layers'], d['first_hidden_layer_scaling_factor'])
    print('all_dimensions:', d['all_dimensions'], flush=True)


    # confirm reporting steps
    print('confirm reporting steps...', flush=True)
    for phase, phase_epochs in [('pretraining', d['pretraining_epochs']), ('finetuning', d['finetuning_epochs']), ('last_layer', d['last_layer_epochs'])]:
        phase_steps = int(phase_epochs/d['batch_fraction'])
        reporting_steps = tsdae_design_functions.create_reporting_steps(phase_steps, d['firstcheckpoint'], d['maxstepspercheckpoint'])
        print('{0}_reporting_steps:'.format(phase), reporting_steps, flush=True)


    # confirm layer training schedule
    print('confirm layer training schedule...', flush=True)
    d['training_schedule'] = tsdae_design_functions.create_layer_training_schedule(d['hidden_layers'], d['pretraining_epochs'], d['finetuning_epochs'], d['last_layer_epochs'], d['use_finetuning'])
    fields = sorted(list(d['training_schedule'][0].keys()))
    for phase in d['training_schedule']:
        print(', '.join(['{0}:{1!s}'.format(field, phase[field]) for field in fields]), flush=True)


    # save design
    print('saving design...', flush=True)
    design_path = '{0}/design.json'.format(d['output_path'])
    print('design_path: {0}'.format(design_path), flush=True)
    with open(design_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(d, fw, indent=2)


    print('done finish_design.', flush=True)
    
    return design_path


if __name__ == '__main__':
    main(sys.argv[1])

