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
import logistic_regression


def main(model_config_path, datamatrix=None):
    
    # load model_config
    print('loading model_config...', flush=False)
    print('model_config_path : {0}'.format(model_config_path), flush=False)
    with open(model_config_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        model_config = json.load(fr)
    
    # set test_index
    if 'test_index' not in model_config:
        model_config['test_index'] = int(model_config['save_folder'].split('/')[1].split('_')[-1]) # expecting save_folder = '{project_folder}/hp_search_{search_id}/hp_combination_{combination_id}'
        print('setting test_index to search_id: {0!s}...'.format(model_config['test_index']), flush=False)
    
    # set valid_index
    if 'valid_index' not in model_config:
        model_config['valid_index'] = int(model_config['save_folder'].split('/')[-1].split('_')[-1]) # expecting save_folder = '{project_folder}/hp_search_{search_id}/hp_combination_{combination_id}'
        print('setting valid_index to combination_id: {0!s}...'.format(model_config['valid_index']), flush=False)
    
    # set datamatrix
    if 'datamatrix' not in model_config:
        model_config['datamatrix'] = datamatrix
        print('setting datamatrix to:', datamatrix, flush=False)
    
    # print model_config
    print('printing model_config...', flush=False)
    for field, value in model_config.items():
        print(field, ':', value, flush=False)
    
    # run analysis
    print('running logistic_regression...', flush=False)
    logistic_regression.main(**model_config)
    
    # done
    print('done train_logistic_regression.py', flush=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Logistic Regression.')
    parser.add_argument('model_config_path', help='path to .json file with configurations for logistic regression', type=str)
    args = parser.parse_args()    
    main(args.model_config_path)
