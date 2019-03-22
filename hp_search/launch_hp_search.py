# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import subprocess
import argparse
import json
import os


def main(search_config_path, job_config_path, model_config_path):
    
    # load search configuration dictionary
    print('loading search configuration...', flush=True)
    print('search_config_path: {0}'.format(search_config_path), flush=True)
    with open(search_config_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        sc = json.loads('\n'.join([x.split('#')[0].rstrip() for x in fr.read().split('\n')]))
    
    # set project_folder, search_folder, and log_path
    print('setting project_folder, search_folder, and log_path...', flush=True)
    project_folder = '/'.join(job_config_path.split('/')[:-1])
    print('project_folder: {0}'.format(project_folder), flush=True)
    search_folders = [x for x in os.listdir(project_folder) if x[:10] == 'hp_search_']
    print('found {0!s} previous hp searches'.format(len(search_folders)), flush=True)
    if len(search_folders) > 0:
        search_ids = [int(x.split('_')[-1]) for x in search_folders]
        search_id = max(search_ids) + 1
    else:
        search_id = 0
    search_folder = '{0}/hp_search_{1!s}'.format(project_folder, search_id)
    os.makedirs(search_folder)
    print('current search_folder: {0}'.format(search_folder), flush=True)
    print('current search_id: {0!s}'.format(search_id), flush=True)
    log_path = '{0}/log.txt'.format(search_folder)
    print('log_path: {0}'.format(log_path), flush=True)
    

    # launch hyperparameter search
    print('launching hyperparameter search...', flush=True)
    search_command = ' '.join([sc['search_environment_path'], sc['search_script_path'], '--search_config_path', search_config_path, '--job_config_path', job_config_path, '--model_config_path', model_config_path, '--search_id', str(search_id)])
    print('search_command: {0}'.format(search_command), flush=True)
    slurm_command = 'sbatch -n 1 -c 1 --mem={0!s}G -t {1!s} -o {2} -e {3} --wrap="{4}"'.format(sc['search_memory'], sc['search_time']*60 + 60, log_path, log_path, search_command)
    print('slurm_command: {0}'.format(slurm_command), flush=True)
    response = subprocess.run(slurm_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    if 'Submitted batch job ' in response:
        job_id = response.strip().split(' ')[-1].strip() # response.replace('Submitted batch job ', '')
    else:
        job_id = ''
    print('job_id: {0}'.format(job_id), flush=True)

    # log slurm jobid
    print('logging slurm job_id to {0}/launch_hp_search.log...'.format(project_folder), flush=True)
    with open('{0}/launched_hp_searches.txt'.format(project_folder), mode='at', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write('\t'.join([job_id, project_folder, search_folder, str(search_id), log_path, search_config_path, job_config_path, model_config_path, search_command, slurm_command]) + '\n')    
            
    # done
    print('done launch_hp_search.py', flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch hyperparameter search.')
    parser.add_argument('--search_config_path', help='path to .json file with configurations for hyperparameter search', type=str)
    parser.add_argument('--job_config_path', help='path to .json file with configurations for jobs', type=str)
    parser.add_argument('--model_config_path', help='path to .json file with configurations for model/analysis', type=str)
    args = parser.parse_args()    
    main(args.search_config_path, args.job_config_path, args.model_config_path)
