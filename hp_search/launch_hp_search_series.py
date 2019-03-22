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
import time


def main(search_config_path, job_config_path, model_config_path, num_repetitions, wait_minutes):
    
    print('search_config_path: {0}'.format(search_config_path), flush=False)
    print('job_config_path: {0}'.format(job_config_path), flush=False)
    print('model_config_path: {0}'.format(model_config_path), flush=False)
    print('num_repetitions: {0!s}'.format(num_repetitions), flush=False)
    print('wait_minutes: {0!s}'.format(wait_minutes), flush=False)
    
    # set project_folder
    print('setting project_folder...', flush=False)
    project_folder = '/'.join(job_config_path.split('/')[:-1])
    print('project_folder: {0}'.format(project_folder), flush=False)
    
    launch_command = ' '.join(['python', 'launch_hp_search.py', '--search_config_path', search_config_path, '--job_config_path', job_config_path, '--model_config_path', model_config_path])
    print('launch_command: {0}'.format(launch_command), flush=False)
    slurm_command = 'sbatch -n 1 -c 1 --mem={0!s}G -t {1!s} -o {2} --wrap="{3}"'.format(8, 5, 'launch_hp_search_instance.log', launch_command)
    print('slurm_command: {0}'.format(slurm_command), flush=False)
    
    for rep in range(num_repetitions):
        
        # submit hyperparameter search launcher
        print('working on repetition {0!s} of {1!s}...'.format(rep, num_repetitions), flush=False)
        response = subprocess.run(slurm_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        if 'Submitted batch job ' in response:
            job_id = response.strip().split(' ')[-1].strip() # response.replace('Submitted batch job ', '')
        else:
            job_id = ''
        print('job_id: {0}'.format(job_id), flush=False)
    
        # log slurm jobid
        print('logging slurm job_id to log_hp_search_series.txt...', flush=False)
        with open('{0}/log_hp_search_series.txt'.format(project_folder), mode='at', encoding='utf-8', errors='surrogateescape') as fw:
            fw.write('\t'.join([str(rep), job_id, search_config_path, job_config_path, model_config_path, launch_command, slurm_command]) + '\n')    
        
        # wait
        time.sleep(60*wait_minutes)
                
    # done
    print('done launch_hp_search_series.py', flush=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch hyperparameter search.')
    parser.add_argument('--search_config_path', help='path to .json file with configurations for hyperparameter search', type=str)
    parser.add_argument('--job_config_path', help='path to .json file with configurations for jobs', type=str)
    parser.add_argument('--model_config_path', help='path to .json file with configurations for model/analysis', type=str)
    parser.add_argument('--num_repetitions', help='number of hyperparameter search repetitions', type=int)
    parser.add_argument('--wait_minutes', help='minutes between hyperparameter search launches', type=float)
    args = parser.parse_args()    
    main(args.search_config_path, args.job_config_path, args.model_config_path, args.num_repetitions, args.wait_minutes)
