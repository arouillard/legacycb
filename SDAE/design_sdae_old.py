# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import json
import pickle
import os
import sdae_functions


def main():
    
    
    # design model
    print('designing model...', flush=True)
    d = {}
    d['previous_hidden_layer'] = 0
    d['previous_finetuning_run'] = 0
    d['pretraining_epochs'] = 100
    d['finetuning_epochs'] = 100
    d['last_layer_epochs'] = 100000
    d['use_finetuning'] = True
    d['min_dimension'] = 2
    d['hidden_layers'] = 5
    d['first_hidden_layer_scaling_factor'] = 'auto' # 'auto' or a scaling factor
    d['noise_probability'] = 1.0
    d['noise_sigma'] = 5.0
    d['noise_distribution'] = 'truncnorm' # 'truncnorm' or 'uniform' sps.truncnorm sps.uniform
    d['noise_operation'] = 'add' # 'add' or 'replace'
    d['initialization_sigma'] = 0.01
    d['initialization_distribution'] = 'truncnorm' # 'truncnorm' tf.truncated_normal
    d['learning_rate'] = 0.001
    d['epsilon'] = 0.001
    d['beta1'] = 0.9
    d['beta2'] = 0.999
    d['batch_fraction'] = 0.1
    d['firstcheckpoint'] = int(1/d['batch_fraction']) # 1, int(1/batch_fraction), or int(pretraining_epochs/batch_fraction)
    d['maxstepspercheckpoint'] = int(1e4)
    d['startsavingstep'] = int(1e4)
    d['include_global_step'] = False
    d['overfitting_score_max'] = 5
    d['activation_function'] = 'relu' # 'relu', 'tanh', 'elu', or 'sigmoid' 
    d['apply_activation_to_output'] = False
    d['processor'] = 'cpu' # 'cpu' or 'gpu'
    d['gpu_memory_fraction'] = 0.2
    d['gpu_id'] = '0'
    d['study_name'] = 'GTEXv6plus'
    d['orientation'] = 'fat' # 'skinny' or 'fat'
    d['input_path'] = 'data/prepared_data/{0}/{1}'.format(d['study_name'], d['orientation'])
    d['output_path'] = 'results/autoencoder/{0}/{1}/hl{2!s}_md{3!s}_fhlsf{4!s}_np{5!s}_ns{6!s}_nd{7}_is{8!s}_id{9}_lr{10!s}_eps{11!s}_bf{12!s}_pte{13!s}_fte{14!s}_uft{15!s}_{16}'\
                  .format(d['study_name'],
                          d['orientation'],
                          d['hidden_layers'],
                          d['min_dimension'],
                          d['first_hidden_layer_scaling_factor'],
                          d['noise_probability'],
                          d['noise_sigma'],
                          d['noise_distribution'],
                          d['initialization_sigma'],
                          d['initialization_distribution'],
                          d['learning_rate'],
                          d['epsilon'],
                          d['batch_fraction'],
                          d['pretraining_epochs'],
                          d['finetuning_epochs'],
                          d['use_finetuning'],
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
    d['all_dimensions'] = sdae_functions.get_layer_dimensions(d['input_dimension'], d['min_dimension'], d['hidden_layers'], d['first_hidden_layer_scaling_factor'])
    print('all_dimensions:', d['all_dimensions'], flush=True)


    # confirm reporting steps
    print('confirm reporting steps...', flush=True)
    for phase, phase_epochs in [('pretraining', d['pretraining_epochs']), ('finetuning', d['finetuning_epochs']), ('last_layer', d['last_layer_epochs'])]:
        phase_steps = int(phase_epochs/d['batch_fraction'])
        reporting_steps = sdae_functions.create_reporting_steps(phase_steps, d['firstcheckpoint'], d['maxstepspercheckpoint'])
        print('{0}_reporting_steps:'.format(phase), reporting_steps, flush=True)


    # confirm layer training schedule
    print('confirm layer training schedule...', flush=True)
    d['training_schedule'] = sdae_functions.create_layer_training_schedule(d['hidden_layers'], d['pretraining_epochs'], d['finetuning_epochs'], d['last_layer_epochs'], d['use_finetuning'])
    fields = sorted(list(d['training_schedule'][0].keys()))
    for phase in d['training_schedule']:
        print(', '.join(['{0}:{1!s}'.format(field, phase[field]) for field in fields]), flush=True)


    # save design
    print('saving design...', flush=True)
    design_path = '{0}/design.json'.format(d['output_path'])
    print('design_path: {0}'.format(design_path), flush=True)
    with open(design_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(d, fw, indent=2)


    print('done.', flush=True)
    
    return design_path

if __name__ == '__main__':
    main()
