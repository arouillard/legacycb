# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import os
import sys
import json
import sdae_functions


def main(design_path, processor='bydesign', gpu_memory_fraction='bydesign', gpu_id='bydesign'):
    
    # load design
    # d is a dictionary containing the auto-encoder design specifications and training phase specifications
    print('loading design...', flush=True)
    print('design_path: {0}'.format(design_path), flush=True)
    with open(design_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        d = json.load(fr)


    # select processor
    print('selecting processor...', flush=True)
    if processor.lower() == 'cpu' or processor.lower() == 'gpu':
        d['processor'] = processor.lower()
    if gpu_memory_fraction != 'bydesign':
        d['gpu_memory_fraction'] = float(gpu_memory_fraction)
    if gpu_id != 'bydesign':
        d['gpu_id'] = gpu_id
    if d['processor'] == 'gpu':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(d['gpu_id'])
        print('processor: {0}, gpu_memory_fraction: {1!s}, gpu_id: {2!s}'.format(d['processor'], d['gpu_memory_fraction'], d['gpu_id']), flush=True)
    else:
        print('processor: {0}'.format(d['processor']), flush=True)

        
    # train model
    print('training model...\n\n', flush=True)
    for phase in d['training_schedule']:
        d['current_hidden_layer'] = phase['hidden_layer']
        d['current_finetuning_run'] = phase['finetuning_run']
        d['current_epochs'] = phase['epochs']
        print('working on hidden layer {0!s}, finetuning run {1!s}, epochs {2!s}...'.format(d['current_hidden_layer'], d['current_finetuning_run'], d['current_epochs']), flush=True)
        d['previous_hidden_layer'], d['previous_finetuning_run'], d['previous_epochs'] = sdae_functions.main(d)
        print('finished hidden layer {0!s}, finetuning run {1!s}, epochs {2!s}.\n\n'.format(d['previous_hidden_layer'], d['previous_finetuning_run'], d['previous_epochs']), flush=True)

        
    print('done.', flush=True)
    
    
if __name__ == '__main__':
    main(*sys.argv[1:])

