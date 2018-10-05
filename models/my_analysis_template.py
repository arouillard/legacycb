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


def main(model_config_path):
    
    # load model_config
    print('loading model_config...', flush=True)
    print('model_config_path : {0}'.format(model_config_path), flush=True)
    with open(model_config_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        model_config = json.load(fr)
    
    # print model_config
    print('printing model_config...', flush=True)
    for field, value in model_config.items():
        print(field, ':', value, flush=True)
    
    # get save_folder
    print('getting save_folder...', flush=True)
    save_folder = model_config['save_folder']
    del model_config['save_folder'] # don't delete if you prefer to keep save_folder in model_config
    print('save_folder : {0}'.format(save_folder), flush=True)
    
    # run analysis
    print('running my analysis...', flush=True)
    # commands to run your analysis with the variable settings in model_config
    # perhaps something like:
    # import my_analysis_function
    # my_analysis_function.main(**model_config)
    
    # get objective function(s) value(s)
    print('getting objective function(s) value(s)...', flush=True)
    # commands to get objective function value/performance score from analysis
    # objective = ...
    objective = 100
    print('objective function value(s) :', objective, flush=True)
    
    # save objective function(s) value(s)
    print('saving objective function(s) value(s)...', flush=True)
    output_path = '{0}/output.json'.format(save_folder)
    print('output_path : {0}'.format(output_path), flush=True)
    with open(output_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(objective, fw, indent=2)
    
    # done
    print('done my_analysis_template.py', flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My analysis template.')
    parser.add_argument('model_config_path', help='path to .json file with configurations for my analysis', type=str)
    args = parser.parse_args()    
    main(args.model_config_path)
