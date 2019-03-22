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
import os


def main(design_list_or_path, processor='gpu', gpu_memory_fraction='bydesign', slurm_partition='us_hpc', slurm_memory='24G', slurm_time='720', skip_existing=True):
    
    # resolve design paths
    print('resolving design paths...', flush=True)
    if type(design_list_or_path) == list:
        design_paths = design_list_or_path
    elif '.json' in design_list_or_path:
        design_paths = [design_list_or_path]
    elif '.txt' in design_list_or_path:
        with open(design_list_or_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            design_paths = fr.read().split('\n')
    else:
        raise ValueError('invalid input to design_list_or_path')
    print('found {0!s} designs...'.format(len(design_paths)), flush=True)
    for design_path in design_paths:
        print('    {0}'.format(design_path), flush=True)
        
    
    # launch jobs
    print('launching jobs...', flush=True)
    for design_path in design_paths:
        print('    working on {0}...'.format(design_path), flush=True)
        job_path = design_path.replace('design.json','job.sh')
        log_path = design_path.replace('design.json','log.txt')
        
        if skip_existing and os.path.exists(log_path):
            # if desired (default True), skip jobs that have already been executed
            print('job has already been executed. skipping...', flush=True)
        
        else:
            # train on GPUs
            if processor.lower() == 'gpu':
                with open(job_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
                    fw.write('#!/bin/sh\n')
                    fw.write('python train_tsdae.py {0} {1} {2}\n'.format(design_path, processor, gpu_memory_fraction))
                subprocess_command = 'sbatch --partition={0} --gres=gpu:1 --mem={1} -t {2} -o {3} {4}'.format(slurm_partition, slurm_memory, slurm_time, log_path, job_path)
            
            # train on CPUs (NOT TESTED!)
            elif processor.lower() == 'cpu':
                with open(job_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
                    fw.write('#!/bin/sh\n')
                    fw.write('source activate python3_tensorflow_cpu\n')
                    fw.write('python train_tsdae.py {0} {1} {2}\n'.format(design_path, processor, gpu_memory_fraction))
                    fw.write('source deactivate python3_tensorflow_cpu\n')
                subprocess_command = 'sbatch -n 1 -c 8 --mem={0} -t {1} -o {2} {3}'.format(slurm_memory, slurm_time, log_path, job_path)
                
            else:
                raise ValueError('invalid processor, specify "cpu" or "gpu"')
                
            print('    submitting "{0}"'.format(subprocess_command), flush=True)
    #        subprocess.run('chmod 777 {0}'.format(job_path), shell=True)
            subprocess.run(subprocess_command, shell=True)
            time.sleep(15)
            
            
    print('done launch_tsdae.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch training of one or more models.')
    parser.add_argument('design_list_or_path', help='path to .json file for a single design or path to .txt file containing paths for a batch of designs', type=str)
    parser.add_argument('--processor', help='specify GPU or CPU training', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--gpu_memory_fraction', help='fraction of GPU memory to reserve for each model, 1 corresponds to 16G, default=get value from each model design.json', type=str, default='bydesign')
    parser.add_argument('--slurm_partition', help='partition for SLURM (queuing system) to assign job, CMS partition is up-gpu and HPC partition is us_hpc', type=str, default='us_hpc')
    parser.add_argument('--slurm_memory', help='memory for SLURM (queuing system) to allocate for job', type=str, default='24G')
    parser.add_argument('--slurm_time', help='time (in minutes) for SLURM (queuing system) to allocate for job', type=str, default='720')
    parser.add_argument('--skip_existing', help='skip jobs that have already been executed (have a log.txt file)', type=int, default=1, choices=[1, 0])
    args = parser.parse_args()
    main(args.design_list_or_path, args.processor, args.gpu_memory_fraction, args.slurm_partition, args.slurm_memory, args.slurm_time, bool(args.skip_existing))

