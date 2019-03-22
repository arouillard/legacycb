# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import subprocess
import sys
import time
import design_base_sdae
import design_sdae_hyperparameter_search


def main(processor='gpu', gpu_memory_fraction='bydesign', hyperparameter_search_type='none', design_paths_path='none', slurm_memory='24G', slurm_time='720'):
    
    if design_paths_path == 'none':
        print('DESIGNING BASE SDAE', flush=True)
        base_design_path, design_paths_path = design_base_sdae.main()
        
        if hyperparameter_search_type == 'none':
            print('SKIPPING SDAE HYPERPARAMETER SEARCH', flush=True)
            design_paths = [base_design_path]
            
        else:
            print('DESIGNING SDAE HYPERPARAMETER SEARCH', flush=True)
            design_paths, design_paths_path = design_sdae_hyperparameter_search.main(base_design_path, hyperparameter_search_type)
    
    else:
        print('LOADING DESIGN PATHS', flush=True)
        with open(design_paths_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            design_paths = fr.read().split('\n')
        
    
    print('LAUNCHING JOBS', flush=True)
    for design_path in design_paths:
        print('    WORKING ON {0}'.format(design_path), flush=True)
        job_path = design_path.replace('design.json','job.sh')
        log_path = design_path.replace('design.json','log.txt')
        
        if processor.lower() == 'gpu':
            with open(job_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
                fw.write('#!/bin/sh\n')
                fw.write('python train_sdae.py {0} {1} {2}\n'.format(design_path, processor, gpu_memory_fraction))
            subprocess_command = 'sbatch --gres=gpu:1 --mem={0} -t {1} -o {2} {3}'.format(slurm_memory, slurm_time, log_path, job_path)
            
        elif processor.lower() == 'cpu':
            with open(job_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
                fw.write('#!/bin/sh\n')
                fw.write('source activate python3_tensorflow_cpu\n')
                fw.write('python train_sdae.py {0} {1} {2}\n'.format(design_path, processor, gpu_memory_fraction))
                fw.write('source deactivate python3_tensorflow_cpu\n')
            subprocess_command = 'sbatch -n 1 -c 8 --mem={0} -t {1} -o {2} {3}'.format(slurm_memory, slurm_time, log_path, job_path)
            
        else:
            raise ValueError('invalid processor, specify "cpu" or "gpu"')
            
        print('    SUBMITTING "{0}"'.format(subprocess_command), flush=True)
#        subprocess.run('chmod 777 {0}'.format(job_path), shell=True)
        subprocess.run(subprocess_command, shell=True)
        time.sleep(10)
    
    
if __name__ == '__main__':
    main(*sys.argv[1:])

