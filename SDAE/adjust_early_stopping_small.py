# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import argparse
import pickle
import json
import os
import numpy as np
import sdae_apply_functions
import datasetIO
import shutil
import matplotlib.pyplot as plt


def main(adjustments_path):
    
    # read adjustments
    print('reading adjustments...', flush=True)
    designpath_selectedstep = {}
    with open(adjustments_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        for line in fr:
            design_path, selected_step = [x.strip() for x in line.split('\t')]
            designpath_selectedstep[design_path] = int(selected_step)
    print('found {0!s} adjustments...'.format(len(designpath_selectedstep)), flush=True)
    
    # make adjustments
    print('making adjustments...', flush=True)
    for didx, (design_path, selected_step) in enumerate(designpath_selectedstep.items()):
        print('working on {0}...'.format(design_path), flush=True)
        print('selected step:{0!s}...'.format(selected_step), flush=True)
        
        
        # load design
        print('loading design...', flush=True)
        with open(design_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            d = json.load(fr)
        if 'apply_activation_to_embedding' not in d: # for legacy code
            d['apply_activation_to_embedding'] = True
        if 'use_batchnorm' not in d: # for legacy code
            d['use_batchnorm'] = False
        if 'skip_layerwise_training' not in d: # for legacy code
            d['skip_layerwise_training'] = False
        phase = d['training_schedule'][-1]
        d['current_hidden_layer'] = phase['hidden_layer']
        d['current_finetuning_run'] = phase['finetuning_run']
        d['current_epochs'] = phase['epochs']
        
        
        # load data
        if didx == 0:
            print('loading data...', flush=True)
            partitions = ['train', 'valid', 'test']
            dataset = {}
            for partition in partitions:
                if partition == 'train':
                    dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], 'valid'))
                    dataset[partition].append(datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], 'test')), 0)
                elif partition == 'valid':
                    dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], 'train'))
                else:
                    dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], partition))

        
        # finish configuration
        print('finishing configuration...', flush=True)
        
        # specify activation function
        if d['activation_function'] == 'tanh':
            activation_function = {'np':sdae_apply_functions.tanh}
        elif d['activation_function'] == 'relu':
            activation_function = {'np':sdae_apply_functions.relu}
        elif d['activation_function'] == 'elu':
            activation_function = {'np':sdae_apply_functions.elu}
        elif d['activation_function'] == 'sigmoid':
            activation_function = {'np':sdae_apply_functions.sigmoid}
    
        # initialize model architecture (number of layers and dimension of each layer)
        d['current_dimensions'] = d['all_dimensions'][:d['current_hidden_layer']+1] # dimensions of model up to current depth
        
        # specify embedding function for current training phase
        # we want the option of skipping the embedding activation function to apply only to the full model
        if not d['apply_activation_to_embedding'] and d['current_dimensions'] == d['all_dimensions']:
            d['current_apply_activation_to_embedding'] = False
        else:
            d['current_apply_activation_to_embedding'] = True

        # specify rows and columns of figure showing data reconstructions
        d['reconstruction_rows'] = int(np.round(np.sqrt(np.min([100, dataset['valid'].shape[0]])/2)))
        d['reconstruction_cols'] = 2*d['reconstruction_rows']
        
        
        # move files
        print('moving files...', flush=True)
        if os.path.exists('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step)):
            if os.path.exists('{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'])):
                shutil.move('{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']),
                            '{0}/variables_layer{1!s}_finetuning{2!s}_old.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
            shutil.copyfile('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                            '{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
        else:
            print('variables do no exist for selected step! skipping...', flush=True)
            continue
        if d['use_batchnorm']:
            if os.path.exists('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step)):
                if os.path.exists('{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'])):
                    shutil.move('{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']),
                                '{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}_old.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
                shutil.copyfile('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step),
                                '{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']))
            else:
                print('batchnorm variables do no exist for selected step! skipping...', flush=True)
                continue
            
        
        # load model variables
        print('loading model variables...', flush=True)
        with open('{0}/variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
            W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
        if d['use_batchnorm']:
            with open('{0}/batchnorm_variables_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
                batchnorm_variables = pickle.load(fr) # gammas, betas, moving_means, moving_variances
            batchnorm_encode_variables, batchnorm_decode_variables = sdae_apply_functions.align_batchnorm_variables(batchnorm_variables, d['current_apply_activation_to_embedding'], d['apply_activation_to_output'])
        
        
        # load reporting variables
        print('loading reporting variables...', flush=True)
        if os.path.exists('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'])):
            with open('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rb') as fr:
                optimization_path = pickle.load(fr)
            reporting_steps = optimization_path['reporting_steps']
            valid_losses = optimization_path['valid_losses']
            train_losses = optimization_path['train_losses']
            valid_noisy_losses = optimization_path['valid_noisy_losses']
            train_noisy_losses = optimization_path['train_noisy_losses']
        else:
            reporting_steps = np.zeros(0, dtype='int32')
            valid_losses = np.zeros(0, dtype='float32')
            train_losses = np.zeros(0, dtype='float32')
            valid_noisy_losses = np.zeros(0, dtype='float32')
            train_noisy_losses = np.zeros(0, dtype='float32')
            with open('{0}/log_layer{1!s}_finetuning{2!s}.txt'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'rt') as fr:
                fr.readline()
                for line in fr:
                    step, train_loss, valid_loss, train_noisy_loss, valid_noisy_loss, time = [float(x.strip()) for x in line.split('\t')]
                    reporting_steps = np.insert(reporting_steps, reporting_steps.size, step)
                    valid_losses = np.insert(valid_losses, valid_losses.size, valid_loss)
                    train_losses = np.insert(train_losses, train_losses.size, train_loss)
                    valid_noisy_losses = np.insert(valid_noisy_losses, valid_noisy_losses.size, valid_noisy_loss)
                    train_noisy_losses = np.insert(train_noisy_losses, train_noisy_losses.size, train_noisy_loss) 
            with open('{0}/optimization_path_layer{1!s}_finetuning{2!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), 'wb') as fw:
                pickle.dump({'reporting_steps':reporting_steps, 'valid_losses':valid_losses, 'train_losses':train_losses, 'valid_noisy_losses':valid_noisy_losses, 'train_noisy_losses':train_noisy_losses}, fw)
        
        
        # compute embedding and reconstruction
        print('computing embedding and reconstruction...', flush=True)        
        recon = {}
        embed = {}
        error = {}
        embed_preactivation = {}
        for partition in partitions:
            if d['use_batchnorm']:
                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
            else:
                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True)
                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
            
            print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)
            
            datasetIO.save_datamatrix('{0}/{1}_embedding_layer{2!s}_finetuning{3!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed[partition])
            datasetIO.save_datamatrix('{0}/{1}_embedding_layer{2!s}_finetuning{3!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed[partition])
            
            if d['current_apply_activation_to_embedding']:
                datasetIO.save_datamatrix('{0}/{1}_embedding_preactivation_layer{2!s}_finetuning{3!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed_preactivation[partition])
                datasetIO.save_datamatrix('{0}/{1}_embedding_preactivation_layer{2!s}_finetuning{3!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run']), embed_preactivation[partition])
                

        # plot loss
        print('plotting loss...', flush=True)
        fg, ax = plt.subplots(1, 1, figsize=(3.25,2.25))
        ax.set_position([0.55/3.25, 0.45/2.25, 2.6/3.25, 1.7/2.25])
        ax.semilogx(reporting_steps, train_losses, ':r', linewidth=1, label='train')
        ax.semilogx(reporting_steps, valid_losses, '-g', linewidth=1, label='valid')
        ax.semilogx(reporting_steps, train_noisy_losses, '--b', linewidth=1, label='train,noisy')
        ax.semilogx(reporting_steps, valid_noisy_losses, '-.k', linewidth=1, label='valid,noisy')
        ax.legend(loc='best', fontsize=8)
        ax.set_ylabel('loss', fontsize=8)
        ax.set_xlabel('steps (selected step:{0!s})'.format(selected_step), fontsize=8)
        ax.set_xlim(reporting_steps[0]-1, reporting_steps[-1]+1)
        # ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', left=True, right=True, bottom=True, top=False, labelleft=True, labelright=False, labelbottom=True, labeltop=False, labelsize=8)
        fg.savefig('{0}/optimization_path_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
        plt.close()


        # plot reconstructions
        print('plotting reconstructions...', flush=True)
        num_recons = min([d['reconstruction_rows']*d['reconstruction_cols'], dataset['valid'].shape[0]])
        x_valid = dataset['valid'].matrix[:num_recons,:]
        xr_valid = recon['valid'].matrix[:num_recons,:]
        if x_valid.shape[1] > 1000:
            x_valid = x_valid[:,:1000]
            xr_valid = xr_valid[:,:1000]
        lb = np.append(x_valid, xr_valid, 1).min(1)
        ub = np.append(x_valid, xr_valid, 1).max(1)
        fg, axs = plt.subplots(d['reconstruction_rows'], d['reconstruction_cols'], figsize=(6.5,3.25))
        for i, ax in enumerate(axs.reshape(-1)):
            if i < num_recons:
                ax.plot(x_valid[i,:], xr_valid[i,:], 'ok', markersize=0.5, markeredgewidth=0)
                ax.set_ylim(lb[i], ub[i])
                ax.set_xlim(lb[i], ub[i])
                ax.tick_params(axis='both', which='major', left=False, right=False, bottom=False, top=False, labelleft=False, labelright=False, labelbottom=False, labeltop=False, pad=4)
                ax.set_frame_on(False)
                ax.axvline(lb[i], linewidth=1, color='k')
                ax.axvline(ub[i], linewidth=1, color='k')
                ax.axhline(lb[i], linewidth=1, color='k')
                ax.axhline(ub[i], linewidth=1, color='k')
            else:
                fg.delaxes(ax)
        fg.savefig('{0}/reconstructions_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=1200)
        plt.close()


        # plot 2d embedding
        if d['current_dimensions'][-1] == 2  and (not d['use_finetuning'] or d['current_finetuning_run'] > 0):
            print('plotting 2d embedding...', flush=True)
            fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
            ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
            ax.plot(embed['train'].matrix[:,0], embed['train'].matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.plot(embed['valid'].matrix[:,0], embed['valid'].matrix[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
            ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False, pad=4)
            ax.set_frame_on(False)
            fg.savefig('{0}/embedding_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
            plt.close()
            
            if d['current_apply_activation_to_embedding']:
                fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
                ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
                ax.plot(embed_preactivation['train'].matrix[:,0], embed_preactivation['train'].matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
                ax.plot(embed_preactivation['valid'].matrix[:,0], embed_preactivation['valid'].matrix[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
                ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False, pad=4)
                ax.set_frame_on(False)
                fg.savefig('{0}/embedding_preactivation_layer{1!s}_finetuning{2!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run']), transparent=True, pad_inches=0, dpi=600)
                plt.close()
        
        
        # log selected step
        with open('{0}/log.txt'.format(d['output_path']), mode='at', buffering=1) as fl:
            fl.write('\nadjusted selected step:{0}\n'.format(selected_step))


    print('done adjust_early_stopping.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adjust early stopping point of one or more models.')
    parser.add_argument('adjustments_path', help='path to .txt file containing design paths and selected step for models to be adjusted', type=str)
    args = parser.parse_args()
    main(args.adjustments_path)

