# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import finish_sdae_design
import json


def main():
    
    
    # design model
    print('designing base model...', flush=True)
    d = {}
    d['previous_hidden_layer'] = 0
    d['previous_finetuning_run'] = 0
    d['pretraining_epochs'] = 100
    d['finetuning_epochs'] = 100
    d['last_layer_epochs'] = 100000
    d['use_finetuning'] = True
    d['min_dimension'] = 2
    d['hidden_layers'] = [5, 4, 6]
    d['first_hidden_layer_scaling_factor'] = [0.1, 0.075, 0.133] # 'auto' or a scaling factor
    d['noise_probability'] = 1.0
    d['noise_sigma'] = [5.0, 2.5, 7.5]
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
    d['maxstepspercheckpoint'] = int(1e5)
    d['startsavingstep'] = int(1e4)
    d['include_global_step'] = False
    d['overfitting_score_max'] = 5
    d['activation_function'] = ['relu', 'tanh'] # 'relu', 'tanh', 'elu', or 'sigmoid' 
    d['apply_activation_to_output'] = False
    d['processor'] = 'gpu' # 'cpu' or 'gpu'
    d['gpu_memory_fraction'] = 0.2
    d['gpu_id'] = '0'
    d['study_name'] = 'GTEXv6'
    d['orientation'] = 'fat' # 'skinny' or 'fat'
    
    with open('design_template.json', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(d, fw, indent=2)
    '''
    # finish design
    print('running finish_sdae_design...', flush=True)
    design_path = finish_sdae_design.main(d)


    # save list of design paths
    print('saving list of design paths...', flush=True)
    design_paths_path = design_path.replace('design.json', 'hyperparameter_search_design_paths.txt')
    print('design_paths_path: {0}'.format(design_paths_path), flush=True)
    with open(design_paths_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write(design_path)
    
    
    print('done design_base_sdae.', flush=True)
    
    return design_path, design_paths_path
    '''

if __name__ == '__main__':
    main()

