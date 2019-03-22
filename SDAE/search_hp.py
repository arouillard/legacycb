# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import sys
import os
import json
import copy
import argparse
from itertools import product
from math import log
import numpy as np
import pickle
import time
import GPy
import GPyOpt
import subprocess


def count_active_jobs(job_ids):
    job_ids_string = ','.join([x for x in job_ids if len(x) > 0])
    if len(job_ids_string) == 0:
        return 0
    else:
        slurm_command = 'squeue -j {0} | wc -l'.format(job_ids_string)
        response = subprocess.run(slurm_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        if 'error' in response:
            return 0
        else:
            return int(response) - 1 # subtract 1 for header line

def submit_job(job_config, model_config_path):
    log_path = '/'.join(model_config_path.split('/')[:-1]) + '/log.txt'
    job_command = ' '.join([job_config['environment_path'], job_config['script_path'], model_config_path])
    print('job_command: {0}'.format(job_command), flush=True)
    if job_config['processor_type'].lower() == 'cpu':
        slurm_command = 'sbatch -n 1 -c {0!s} --mem={1!s}G -t {2!s} -o {3} -e {4} --wrap="{5}"'.format(job_config['num_processors'], job_config['memory'], job_config['time']*60, log_path, log_path, job_command)
    elif job_config['processor_type'].lower() == 'gpu':
        if job_config['cluster'].lower() == 'cms':
            partition = 'up-gpu'
        elif job_config['cluster'].lower() == 'gsktech':
            partition = 'us_hpc'
        else:
            raise ValueError('invalid job_config cluster. must be cms or gsktech.')
        slurm_command = 'sbatch --partition={0} --gres=gpu:{1!s} --mem={2!s}G -t {3!s} -o {4} -e {5} --wrap="{6}"'.format(partition, job_config['num_processors'], job_config['memory'], job_config['time']*60, log_path, log_path, job_command)
    else:
        raise ValueError('invalid job_config processor_type. must be cpu or gpu.')
    print('slurm_command: {0}'.format(slurm_command), flush=True)
    response = subprocess.run(slurm_command, shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
    if 'Submitted batch job ' in response:
        job_id = response.strip().split(' ')[-1] # response.replace('Submitted batch job ', '')
    else:
        job_id = ''
    print('job_id: {0}'.format(job_id), flush=True)
    return job_id

def update_job_info(job_id, combination_id, is_queued, job_ids, job_times):
    is_queued[combination_id] = False
    job_ids[combination_id] = job_id
    job_times[combination_id] = time.time()
    return is_queued, job_ids, job_times

def create_line_search_combinations(base_model_config, hyperparameters, hyperparameter_specs):
    print('creating hyperparameter combinations using line search...', flush=True)
    hyperparameter_combinations = np.zeros((0, len(hyperparameters)), dtype='float64')
    Xbase = np.array([base_model_config['search_domain'][hp] for hp in hyperparameters], dtype='float64').reshape(1,-1)
    # iterate over hyperparameters
    for j, hp in enumerate(hyperparameters):
        Xi = Xbase.copy()
        # iterate over hyperparameter values
        for search_domain_value in hyperparameter_specs[hp]['grid']:
            Xi[0,j] = search_domain_value
            hyperparameter_combinations = np.append(hyperparameter_combinations, Xi.copy(), 0)
    np.random.shuffle(hyperparameter_combinations)
    search_types = np.full(hyperparameter_combinations.shape[0], 'line', dtype='object')
    print('num_hyperparameter_combinations: {0!s}'.format(hyperparameter_combinations.shape[0]), flush=True)
    return hyperparameter_combinations, search_types

def create_grid_search_combinations(hyperparameters, hyperparameter_specs):
    print('creating hyperparameter combinations using grid search...', flush=True)
    hyperparameter_combinations = np.array(list(product(*[hyperparameter_specs[hp]['grid'] for hp in hyperparameters])), dtype='float64')
    np.random.shuffle(hyperparameter_combinations)
    search_types = np.full(hyperparameter_combinations.shape[0], 'grid', dtype='object')
    print('num_hyperparameter_combinations: {0!s}'.format(hyperparameter_combinations.shape[0]), flush=True)
    return hyperparameter_combinations, search_types
    
def create_random_search_combinations(num_combinations, domains):
    print('creating hyperparameter combinations using random search...', flush=True)
    hyperparameter_combinations = np.zeros((num_combinations, len(domains)), dtype='float64')
    for j, hpd in enumerate(domains):
        if hpd['type'].lower() == 'continuous':
            hyperparameter_combinations[:,j] = (hpd['domain'][-1] - hpd['domain'][0])*np.random.rand(num_combinations) + hpd['domain'][0]
        else:
            hyperparameter_combinations[:,j] = np.random.choice(hpd['domain'], num_combinations, replace=True)
    search_types = np.full(hyperparameter_combinations.shape[0], 'random', dtype='object')
    print('num_hyperparameter_combinations: {0!s}'.format(hyperparameter_combinations.shape[0]), flush=True)
    return hyperparameter_combinations, search_types

def create_bopt_search_combinations(X, Y, domains, bopt_config, acquisition_types, model_types, model_estimation_methods, kernels):
    print('creating hyperparameter combinations using bayesian optimization...', flush=True)
    kernel_name_to_func = {'RatQuad':GPy.kern.RatQuad(input_dim=len(domains), ARD=True), 'ExpQuad':GPy.kern.ExpQuad(input_dim=len(domains), ARD=True), 'Matern52':GPy.kern.Matern52(input_dim=len(domains), ARD=True)}
    bopt_config['acquisition_type'] = np.random.choice(acquisition_types, replace=True)
    bopt_config['model_type'] = np.random.choice(model_types, replace=True)
    model_estimation_method = np.random.choice(model_estimation_methods, replace=True)
    if model_estimation_method.lower() == 'mcmc':
        bopt_config['acquisition_type'] += '_MCMC'
        bopt_config['model_type'] += '_MCMC'
    elif model_estimation_method.lower() == 'mle':
        pass
    else:
        raise ValueError('invalid model_estimation_method. must be MCMC or MLE.')
    kernel_name = np.random.choice(kernels, replace=True)
    bopt_config['kernel'] = kernel_name_to_func[kernel_name]
    print('acquisition type: {0}'.format(bopt_config['acquisition_type']), flush=True)
    print('model type: {0}'.format(bopt_config['model_type']), flush=True)
    print('kernel: {0}'.format(kernel_name), flush=True)
    print('fitting {0} {1} model and finding {2} optimum...'.format(bopt_config['model_type'], kernel_name, bopt_config['acquisition_type']), flush=True)
    bopt_start_time = time.time()
    bopt = GPyOpt.methods.BayesianOptimization(f=None, domain=domains, X=X['search_domain'][~np.isnan(Y),:], Y=Y[~np.isnan(Y)].reshape(-1,1), **bopt_config)
    hyperparameter_combinations = bopt.suggest_next_locations(pending_X=X['search_domain'][np.isnan(Y),:])
    search_types = np.full(hyperparameter_combinations.shape[0], kernel_name + '_' + bopt_config['acquisition_type'], dtype='object')
    print(bopt.model.model, flush=True)
    print('bopt iteration time: {0:1.3g} min'.format((time.time() - bopt_start_time)/60.0), flush=True)
    print('num_hyperparameter_combinations: {0!s}'.format(hyperparameter_combinations.shape[0]), flush=True)
    return hyperparameter_combinations, search_types

def create_initial_hyperparameter_combinations(search_type, grid_suggestion_probability, base_model_config, hyperparameters, domains, hyperparameter_specs):
    if search_type.lower() == 'line':
        return create_line_search_combinations(base_model_config, hyperparameters, hyperparameter_specs)
    elif search_type.lower() == 'grid':
        return create_grid_search_combinations(hyperparameters, hyperparameter_specs)
    elif search_type.lower() == 'random':
        return create_random_search_combinations(0, domains)
    elif search_type.lower() == 'bopt':
        hyperparameter_combinations, search_types = create_random_search_combinations(0, domains)
        if grid_suggestion_probability > 0:
            hpc, st = create_grid_search_combinations(hyperparameters, hyperparameter_specs)
            hyperparameter_combinations = np.append(hyperparameter_combinations, hpc, 0)
            search_types = np.append(search_types, st)
        return hyperparameter_combinations, search_types
    else:
        raise ValueError('invalid search_type. must be line, grid, random, or bopt')

def create_model_config(combination_values, hyperparameters, hyperparameter_specs, base_model_config):
    # copy base_model_config
    print('copying base_model_config...', flush=True)
    model_config = copy.deepcopy(base_model_config)
    # update hyperparameters
    for hyperparameter, search_domain_value in zip(hyperparameters, combination_values):
        model_config['search_domain'][hyperparameter] = search_domain_value
        model_config['model_space'][hyperparameter] = hyperparameter_specs[hyperparameter]['search_domain_to_model_space'](search_domain_value)
        print('updating {0} to {1!s}...'.format(hyperparameter, model_config['model_space'][hyperparameter]), flush=True)
    return model_config

def save_model_config(model_config, combination_id, search_folder):
    # create save_folder
    print('creating save_folder...', flush=True)
    combination_folder = '{0}/hp_combination_{1!s}'.format(search_folder, combination_id)
    model_config['search_domain']['save_folder'] = combination_folder
    model_config['model_space']['save_folder'] = combination_folder
    if not os.path.exists(combination_folder):
        os.makedirs(combination_folder)
    print("combination_folder / model_config['save_folder']: {0}".format(combination_folder), flush=True)
    # save model_config
    print('saving model_config...', flush=True)
    model_config_path = '{0}/input.json'.format(model_config['model_space']['save_folder'])
    print('model_config_path: {0}'.format(model_config_path), flush=True)
    with open(model_config_path, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(model_config['model_space'], fw, indent=2)
    return model_config, model_config_path

def append_search_data_arrays(model_config, model_config_path, search_type, combination_id, hyperparameters, X, Y, is_queued, search_types, combination_ids, job_ids, job_times, model_config_paths, model_configs):
    # append hyperparameter combination search data to arrays
    print('appending hyperparameter combination info to search data arrays...', flush=True)
    X['search_domain'] = np.append(X['search_domain'], np.array([model_config['search_domain'][hp] for hp in hyperparameters], dtype='float64').reshape(1,-1), 0)
    X['model_space'] = np.append(X['model_space'], np.array([model_config['model_space'][hp] for hp in hyperparameters], dtype='object').reshape(1,-1), 0)
    Y = np.insert(Y, Y.size, np.nan)
    is_queued = np.insert(is_queued, is_queued.size, True)
    search_types = np.insert(search_types, search_types.size, search_type)
    combination_ids = np.insert(combination_ids, combination_ids.size, combination_id)
    job_ids = np.insert(job_ids, job_ids.size, '')
    job_times = np.insert(job_times, job_times.size, np.nan)
    model_config_paths.append(model_config_path)
    model_configs.append(model_config)
    return X, Y, is_queued, search_types, combination_ids, job_ids, job_times, model_config_paths, model_configs

def update_y(Y, transform_y, job_times, model_configs):
    for i in np.isnan(Y).nonzero()[0]:
        results_path = '{0}/output.json'.format(model_configs[i]['model_space']['save_folder'])
        if os.path.exists(results_path):
            with open(results_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
                results = json.load(fr)
            if type(results) == float:
                objective = results
            elif type(results) == list:
                objective = results[0]
            elif type(results) == dict:
                objective = results['objective']
            else:
                raise ValueError('results must be float, list, or dict.')
            Y[i] = transform_y(objective)
            job_times[i] = (time.time() - job_times[i])/60.0
    return Y, job_times

def main(search_config_path, job_config_path):
    
    
    # load search configuration
    print('loading search configuration...', flush=True)
    print('search_config_path: {0}'.format(search_config_path), flush=True)
    with open(search_config_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        sc = json.loads('\n'.join([x.split('#')[0].rstrip() for x in fr.read().split('\n')]))
    
    if type(sc['bopt_config']['acquisition_type']) == str:
        acquisition_types = [sc['bopt_config']['acquisition_type']]
    else:
        acquisition_types = sc['bopt_config']['acquisition_type']
    del sc['bopt_config']['acquisition_type']
    if type(sc['bopt_config']['model_type']) == str:
        model_types = [sc['bopt_config']['model_type']]
    else:
        model_types = sc['bopt_config']['model_type']
    del sc['bopt_config']['model_type']
    if type(sc['bopt_config']['model_estimation_method']) == str:
        model_estimation_methods = [sc['bopt_config']['model_estimation_method']]
    else:
        model_estimation_methods = sc['bopt_config']['model_estimation_method']
    del sc['bopt_config']['model_estimation_method']
    if type(sc['bopt_config']['kernel']) == str:
        kernels = [sc['bopt_config']['kernel']]
    else:
        kernels = sc['bopt_config']['kernel']
    del sc['bopt_config']['kernel']
    if sc['y_transformation'].lower() == 'neglog':
        transform_y = lambda y: -np.log10(y)
    elif sc['y_transformation'].lower() == 'log':
        transform_y = lambda y: np.log10(y)
    elif sc['y_transformation'].lower() == 'neg':
        transform_y = lambda y: -y
    else:
        transform_y = lambda y: y
    
    
    # load job configuration
    print('loading job configuration...', flush=True)
    print('job_config_path: {0}'.format(job_config_path), flush=True)
    with open(job_config_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        jc = json.loads('\n'.join([x.split('#')[0].rstrip() for x in fr.read().split('\n')]))

    
    # set folders
    print('setting folders...', flush=True)
    project_folder = '/'.join(job_config_path.split('/')[:-1])
    print('project_folder: {0}'.format(project_folder), flush=True)
    search_folders = [x for x in os.listdir(project_folder) if x[:10] == 'hp_search_']
    search_ids = [int(x.split('_')[-1]) for x in search_folders]
    search_id = max(search_ids)
    search_folder = '{0}/hp_search_{1!s}'.format(project_folder, search_id)
    print('current search_folder: {0}'.format(search_folder), flush=True)
    print('current search_id: {0!s}'.format(search_id), flush=True)
    
    
    # create base model_config and collect hyperparameter search specs
    print('creating base model_config and collecting hyperparameter search specs...', flush=True)
    hyperparameter_specs = {}
    hyperparameters = []
    domains = []
    base_model_config = {'search_domain':{}, 'model_space':{}}
    for parameter, specs in jc['model_config'].items():
        if type(specs) == dict:
            
            if specs['type'] == 'categorical':
                model_space_values = specs['domain'].copy()
                specs['domain'] = list(range(len(model_space_values)))
                specs['grid'] = specs['domain'].copy()
                specs['search_domain_to_model_space'] = lambda search_domain_value, sdvs=specs['domain'].copy(), msvs=model_space_values.copy(): {sdv:msv for sdv, msv in zip(sdvs, msvs)}[search_domain_value]
                specs['model_space_to_search_domain'] = lambda model_space_value, sdvs=specs['domain'].copy(), msvs=model_space_values.copy(): {msv:sdv for sdv, msv in zip(sdvs, msvs)}[model_space_value]
                base_model_config['search_domain'][parameter] = specs['domain'][0]
                
            elif specs['type'] == 'discrete' or specs['type'] == 'continuous':
                model_space_values = sorted(specs['domain'])
                model_space_min = model_space_values[0]
                model_space_max = model_space_values[-1]
                model_space_mid = (model_space_min + model_space_max)/2
                if specs['transformation'] == 'linear':
                    specs['search_domain_to_model_space'] = lambda search_domain_value, msmax=model_space_max, msmin=model_space_min: (msmax - msmin)*search_domain_value + msmin
                    specs['model_space_to_search_domain'] = lambda model_space_value, msmax=model_space_max, msmin=model_space_min: (model_space_value - msmin)/(msmax - msmin)
                elif specs['transformation'] == 'log':
                    specs['search_domain_to_model_space'] = lambda search_domain_value, msmax=model_space_max, msmin=model_space_min: msmin*(msmax/msmin)**search_domain_value
                    specs['model_space_to_search_domain'] = lambda model_space_value, msmax=model_space_max, msmin=model_space_min: log(model_space_value/msmin, msmax/msmin)
                else:
                    raise ValueError('invalid hyperparameter transformation. must be linear or log.')
                specs['domain'] = [specs['model_space_to_search_domain'](msv) for msv in model_space_values]
                
                if specs['type'] == 'discrete':
                    if sc['num_grid_points_per_hyperparameter'] < 2:
                        specs['grid'] = [specs['model_space_to_search_domain'](model_space_values[np.argmin([(msv - model_space_mid)**2 for msv in model_space_values])])]
                    elif len(specs['domain']) <= sc['num_grid_points_per_hyperparameter']:
                        specs['grid'] = specs['domain'].copy()
                    else:
                        interval = round(len(specs['domain'])/sc['num_grid_points_per_hyperparameter'])
                        specs['grid'] = specs['domain'][::interval]
                        if len(specs['grid']) < sc['num_grid_points_per_hyperparameter']:
                            specs['grid'].append(specs['domain'][-1])
                        elif len(specs['grid']) == sc['num_grid_points_per_hyperparameter']:
                            specs['grid'][-1] = specs['domain'][-1]
                        else:
                            while len(specs['grid']) > sc['num_grid_points_per_hyperparameter']:
                                del(specs['grid'][-2])
                                
                else:
                    if sc['num_grid_points_per_hyperparameter'] < 2:
                        specs['grid'] = [specs['model_space_to_search_domain'](model_space_mid)]
                    else:
                        specs['grid'] = np.linspace(specs['domain'][0], specs['domain'][-1], sc['num_grid_points_per_hyperparameter'])
                        
                if len(specs['grid']) < 3:
                    base_model_config['search_domain'][parameter] = specs['grid'][0]           
                else:
                    base_model_config['search_domain'][parameter] = specs['grid'][round(len(specs['grid'])/2)-1]
                
            else:
                raise ValueError('invalid hyperparameter type. must be categorical, discrete, or continuous.')
                
            base_model_config['model_space'][parameter] = specs['search_domain_to_model_space'](base_model_config['search_domain'][parameter])   
            hyperparameter_specs[parameter] = copy.deepcopy(specs)
            hyperparameters.append(parameter)
            domains.append({'name':parameter, 'type':specs['type'], 'domain':specs['domain'].copy()})
            
        else:
            base_model_config['model_space'][parameter] = specs
            base_model_config['search_domain'][parameter] = specs


    # create hyperparameter variations
    X = {'search_domain':np.zeros((0, len(hyperparameters)), dtype='float64'), 'model_space':np.zeros((0, len(hyperparameters)), dtype='object')}
    Y = np.zeros(0, dtype='float64')
    is_queued = np.ones(0, dtype='bool')
    search_types = np.zeros(0, dtype='object')
    combination_ids = np.zeros(0, dtype='int64')
    job_ids = np.zeros(0, dtype='object')
    job_times = np.zeros(0, dtype='float64')
    model_config_paths = []
    model_configs = []
    
    # create initial hyperparameter combinations
    print('creating initial hyperparameter combinations...', flush=True)
    hyperparameter_combinations, hyperparameter_search_types = create_initial_hyperparameter_combinations(sc['search_type'], sc['grid_suggestion_probability'], base_model_config, hyperparameters, domains, hyperparameter_specs)
    
    # iterate over hyperparameter combinations
    for combination_id, (combination_values, search_type) in enumerate(zip(hyperparameter_combinations, hyperparameter_search_types)):
        print('working on hyperparameter combination {0!s}...'.format(combination_id), flush=True)
        
        # create model_config
        print('creating model_config...', flush=True)
        mc = create_model_config(combination_values, hyperparameters, hyperparameter_specs, base_model_config)
        
        # create save_folder and save model_config
        print('creating save_folder and saving model_config...', flush=True)
        mc, mc_path = save_model_config(mc, combination_id, search_folder)
        
        # update search data arrays
        print('updating search data arrays...', flush=True)
        X, Y, is_queued, search_types, combination_ids, job_ids, job_times, model_config_paths, model_configs = append_search_data_arrays(mc, mc_path, search_type, combination_id, hyperparameters, X, Y, is_queued, search_types, combination_ids, job_ids, job_times, model_config_paths, model_configs)
    
    # save initial search data
    print('saving initial search data...', flush=True)
    search_data_path = '{0}/hp_search_data.pickle'.format(search_folder)
    with open(search_data_path, 'wb') as fw:
        pickle.dump({'X':X, 'Y':Y, 'is_queued':is_queued, 'search_types':search_types, 'combination_ids':combination_ids, 'job_ids':job_ids, 'model_config_paths':model_config_paths, 'model_configs':model_configs}, fw)




    # run search
    print('running search...', flush=True)
    start_time = time.time()
    elapsed_time = (time.time() - start_time)/3600.0
    num_suggestions = 0
    num_iterations = 0
    prev_num_completed_combinations = 0
    max_consecutive_bopt_failures = 5
    num_consecutive_bopt_failures = 0
    z_headers = ['iterations', 'suggestions', 'time', 'completions', 'minimum', 'index', 'job_time', 'search_type'] + hyperparameters
    Z = []
    while (elapsed_time < sc['search_time'] and num_suggestions < sc['max_suggestions'] and num_consecutive_bopt_failures < max_consecutive_bopt_failures and (sc['search_type'] == 'bopt' or sc['search_type'] == 'random')) or is_queued.any():
        num_active_combinations = count_active_jobs(job_ids)
        print('ACTIVE COMBINATIONS: {0!s}'.format(num_active_combinations), flush=True)
        Y, job_times = update_y(Y, transform_y, job_times, model_configs)
        num_completed_combinations = (~np.isnan(Y)).sum()
        print('COMPLETED COMBINATIONS: {0!s}'.format(num_completed_combinations), flush=True)
        
        if num_completed_combinations > prev_num_completed_combinations:
            prev_num_completed_combinations = num_completed_combinations
            i_min = np.nanargmin(Y)
            print('MINIMUM LOSS: {0!s}'.format(Y[i_min]), flush=True)
            Z.append([num_iterations, num_suggestions, elapsed_time, num_completed_combinations, Y[i_min], i_min, job_times[i_min], search_types[i_min]] + X['model_space'][i_min,:].tolist())
            print({k:v for k,v in zip(z_headers, Z[-1])}, flush=True)
        
        suggest_grid_point = (np.random.rand() < sc['grid_suggestion_probability']) or (sc['grid_suggestion_probability'] == 1) or sc['search_type'] == 'line' or sc['search_type'] == 'grid'
        if num_active_combinations >= sc['max_active_points']:
            print('REACHED MAX ACTIVE COMBINATIONS: {0!s}. WAITING...'.format(sc['max_active_points']), flush=True)
            time.sleep(55)
            
        elif is_queued.any() and (suggest_grid_point or elapsed_time > sc['search_time'] or num_suggestions > sc['max_suggestions']):
            combination_id = is_queued.nonzero()[0][0]
            print('SUBMITTING QUEUED GRID OR LINE COMBINATION {0!s}...'.format(combination_id), flush=True)
            job_id = submit_job(jc, model_config_paths[combination_id])
            is_queued, job_ids, job_times = update_job_info(job_id, combination_id, is_queued, job_ids, job_times)
            num_suggestions += 1
            
        else:
            suggest_random_point = (np.random.rand() < sc['random_suggestion_probability']) or (sc['random_suggestion_probability'] == 1) or sc['search_type'] == 'random'
            if suggest_random_point or num_completed_combinations < sc['min_initial_points']:
                print('CREATING NEW RANDOM COMBINATION...', flush=True)
                hyperparameter_combinations, hyperparameter_search_types = create_random_search_combinations(1, domains)
            else:
                print('CREATING NEW BOPT COMBINATION...', flush=True)
                try:
                    hyperparameter_combinations, hyperparameter_search_types = create_bopt_search_combinations(X, Y, domains, sc['bopt_config'], acquisition_types, model_types, model_estimation_methods, kernels)
                    num_consecutive_bopt_failures = 0
                except:
                    bopt_error = sys.exc_info()
                    num_consecutive_bopt_failures += 1
                    print('BOPT ERROR: {0} {1}'.format(bopt_error[0], bopt_error[1]), flush=True)
                    print('num_consecutive_bopt_failures: {0!s}'.format(num_consecutive_bopt_failures), flush=True)                    
                    hyperparameter_combinations = np.zeros((0, len(hyperparameters)), dtype='float64')
                    hyperparameter_search_types = np.zeros(0, dtype='object')
                    
            # iterate over hyperparameter combinations
            for combination_values, search_type in zip(hyperparameter_combinations, hyperparameter_search_types):
                combination_id = combination_ids.size
                print('working on hyperparameter combination {0!s}...'.format(combination_id), flush=True)
            
                # create model_config
                print('creating model_config...', flush=True)
                mc = create_model_config(combination_values, hyperparameters, hyperparameter_specs, base_model_config)
                
                # create save_folder and save model_config
                print('creating save_folder and saving model_config...', flush=True)
                mc, mc_path = save_model_config(mc, combination_id, search_folder)
                
                # update search data arrays
                print('updating search data arrays...', flush=True)
                X, Y, is_queued, search_types, combination_ids, job_ids, job_times, model_config_paths, model_configs = append_search_data_arrays(mc, mc_path, search_type, combination_id, hyperparameters, X, Y, is_queued, search_types, combination_ids, job_ids, job_times, model_config_paths, model_configs)
        
                print('SUBMITTING NEW COMBINATION {0!s}...'.format(combination_id), flush=True)
                job_id = submit_job(jc, model_config_paths[combination_id])
                is_queued, job_ids, job_times = update_job_info(job_id, combination_id, is_queued, job_ids, job_times)
                num_suggestions += 1
        
            # save search data
            print('saving search data...', flush=True)
            with open(search_data_path, 'wb') as fw:
                pickle.dump({'X':X, 'Y':Y, 'is_queued':is_queued, 'search_types':search_types, 'combination_ids':combination_ids, 'job_ids':job_ids, 'job_times':job_times, 'model_config_paths':model_config_paths, 'model_configs':model_configs}, fw)
        
        time.sleep(5)
        elapsed_time = (time.time() - start_time)/3600.0
        num_iterations += 1

    
    # wait for jobs to finish
    num_active_combinations = count_active_jobs(job_ids)
    while num_active_combinations > 0 and elapsed_time < (sc['search_time'] + 1 - 600/3600):
        print('ACTIVE COMBINATIONS: {0!s}. WAITING FOR JOBS TO FINISH...'.format(num_active_combinations), flush=True)
        time.sleep(300)
        num_active_combinations = count_active_jobs(job_ids)
        elapsed_time = (time.time() - start_time)/3600.0
    
    # final search data collection
    print('collecting final results...', flush=True)
    Y, job_times = update_y(Y, transform_y, job_times, model_configs)
    
    # save search data
    print('saving search data...', flush=True)
    with open(search_data_path, 'wb') as fw:
        pickle.dump({'X':X, 'Y':Y, 'is_queued':is_queued, 'search_types':search_types, 'combination_ids':combination_ids, 'job_ids':job_ids, 'model_config_paths':model_config_paths, 'model_configs':model_configs}, fw)


    print('done search_hp.py', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search hyperparameter space according to search and job configurations.')
    parser.add_argument('--search_config_path', help='path to .json file with configurations for hyperparameter search', type=str)
    parser.add_argument('--job_config_path', help='path to .json file with configurations for jobs', type=str)
    args = parser.parse_args()    
    main(args.search_config_path, args.job_config_path)

