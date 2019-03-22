# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import argparse
import json
import os
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
        
    
    # collect model results
    print('collecting model results...', flush=True)
    if '.json' in design_list_or_path:
        base_path = '/'.join(design_paths[0].split('/')[:-1])
    else:
        base_path = '/'.join(design_paths[0].split('/')[:-2])
    figure_dirs = {'embedding':base_path + '/figure_collection{0}/embedding'.format(version),
                   'embedding_preactivation':base_path + '/figure_collection{0}/embedding_preactivation'.format(version),
                   'loglog_optimization_path':base_path + '/figure_collection{0}/loglog_optimization_path'.format(version),
                   'optimization_path':base_path + '/figure_collection{0}/optimization_path'.format(version),
                   'auprc_per_col':base_path + '/figure_collection{0}/auprc_per_col'.format(version),
                   'auprc_per_row':base_path + '/figure_collection{0}/auprc_per_row'.format(version),
                   'auroc_per_col':base_path + '/figure_collection{0}/auroc_per_col'.format(version),
                   'auroc_per_row':base_path + '/figure_collection{0}/auroc_per_row'.format(version),
                   'reconstructions':base_path + '/figure_collection{0}/reconstructions'.format(version),
                   'proj2d':base_path + '/figure_collection{0}/proj2d'.format(version),
                   'proj2d_preactivation':base_path + '/figure_collection{0}/proj2d_preactivation'.format(version)}
#                   'intermediate_embedding':base_path + '/figure_collection{0}/intermediate_embedding'.format(version),
#                   'intermediate_embedding_preactivation':base_path + '/figure_collection{0}/intermediate_embedding_preactivation'.format(version),
#                   'intermediate_reconstructions':base_path + '/figure_collection{0}/intermediate_reconstructions'.format(version),
#                   'intermediate_embedding':base_path + '/figure_collection{0}/intermediate_embedding'.format(version),
#                   'intermediate_embedding_preactivation':base_path + '/figure_collection{0}/intermediate_embedding_preactivation'.format(version)}
    for fp in figure_dirs.values():
        if not os.path.exists(fp):
            os.makedirs(fp)

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
        
        target_label_condensed = design['output_path'].split('/')[-1].replace('_', '')
        target_label_condensed = target_label_condensed.replace('sigmoid', 'S').replace('tanh', 'T').replace('relu', 'R').replace('truncnorm', 'TN').replace('uniform', 'U').replace('bernoulli', 'B').replace('replace', 'R').replace('add', 'A').replace('flip', 'F').replace('True', 'T').replace('False', 'F')
        
        figures = [x for x in os.listdir(design['output_path']) if x[-4:] == '.png' and 'layer{0}_finetuning1'.format(design['hidden_layers']) in x]
        for figure in figures:
            source_path = '{0}/{1}'.format(design['output_path'], figure)
            if 'step' in figure:
                figure_suffix = '_s' + figure.split('step')[-1].replace('_coloredby', 'cb').replace('_general_tissue', 'GT').replace('_train', 'trn').replace('_test', 'tst').replace('_valid', 'vld')
            else:
                figure_suffix = '.png'
            figure_dir = ''
            for figure_prefix in ['embedding', 'embedding_preactivation', 'proj2d', 'proj2d_preactivation', 'loglog_optimization_path', 'optimization_path', 'reconstructions', 'auprc_per_row', 'auprc_per_col', 'auroc_per_row', 'auroc_per_col']:
                if figure_prefix in figure:
                    figure_dir = figure_dirs[figure_prefix]
                    break
            target_path = '{0}/{1}{2}'.format(figure_dir, target_label_condensed, figure_suffix)
            if os.path.exists(source_path):
                shutil.copyfile(source_path, target_path)
                    
                


    print('done collect_figures.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect figures from one or more models.')
    parser.add_argument('design_list_or_path', help='path to .json file for a single design or path to .txt file containing paths for a batch of designs', type=str)
    args = parser.parse_args()
    main(args.design_list_or_path)

