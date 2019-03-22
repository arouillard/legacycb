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
import copy
import numpy as np
import sdae_apply_functions_7_ccramp as sdae_apply_functions
import datasetIO
import matplotlib.pyplot as plt


def main(visualizations_path):
    
    # read visualizations
    print('reading visualizations...', flush=True)
    designpath_selectedstep = {}
    with open(visualizations_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        for line in fr:
            design_path, selected_step = [x.strip() for x in line.split('\t')]
            designpath_selectedstep[design_path] = int(selected_step)
    print('found {0!s} visualizations...'.format(len(designpath_selectedstep)), flush=True)
    
    # make visualizations
    print('making visualizations...', flush=True)
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
#        if not d['apply_activation_to_embedding'] and d['current_dimensions'] == d['all_dimensions']:
#            d['current_apply_activation_to_embedding'] = False
#        else:
#            d['current_apply_activation_to_embedding'] = True
        if d['current_dimensions'] == d['all_dimensions']:
            if d['apply_activation_to_embedding']:
                d['current_apply_activation_to_embedding'] = True
                use_softmax = True
            else:
                d['current_apply_activation_to_embedding'] = False
                use_softmax = False
        else:
            d['current_apply_activation_to_embedding'] = True
            use_softmax = False
        print('current_apply_activation_to_embedding: {0!s}'.format(d['current_apply_activation_to_embedding']), flush=True)
        print('use_softmax: {0!s}'.format(use_softmax), flush=True)

        # specify rows and columns of figure showing data reconstructions
        d['reconstruction_rows'] = int(np.round(np.sqrt(np.min([100, dataset['valid'].shape[0]])/2)))
        d['reconstruction_cols'] = 2*d['reconstruction_rows']
        
        # load model variables
        print('loading model variables...', flush=True)
        with open('{0}/intermediate_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), 'rb') as fr:
            W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
        if d['use_batchnorm']:
            with open('{0}/intermediate_batchnorm_variables_layer{1!s}_finetuning{2!s}_step{3!s}.pickle'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), 'rb') as fr:
                batchnorm_variables = pickle.load(fr) # gammas, betas, moving_means, moving_variances
            batchnorm_encode_variables, batchnorm_decode_variables = sdae_apply_functions.align_batchnorm_variables(batchnorm_variables, d['current_apply_activation_to_embedding'], d['apply_activation_to_output'])
        
        # compute embedding and reconstruction
        print('computing embedding and reconstruction...', flush=True)        
        recon = {}
        embed = {}
        error = {}
        embed_preactivation = {}
        for partition in partitions:
            if d['use_batchnorm']:
#                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
#                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, bn_variables=batchnorm_encode_variables)
                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], use_softmax, d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True, bn_encode_variables=batchnorm_encode_variables, bn_decode_variables=batchnorm_decode_variables)
                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, use_softmax=use_softmax, bn_variables=batchnorm_encode_variables)
            else:
#                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True)
#                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False)
                recon[partition], embed[partition], error[partition] = sdae_apply_functions.encode_and_decode(dataset[partition], W, Be, Bd, activation_function['np'], d['current_apply_activation_to_embedding'], use_softmax, d['apply_activation_to_output'], return_embedding=True, return_reconstruction_error=True)
                embed_preactivation[partition] = sdae_apply_functions.encode(dataset[partition], W, Be, activation_function['np'], apply_activation_to_embedding=False, use_softmax=use_softmax)
            
            print('{0} reconstruction error: {1:1.3g}'.format(partition, error[partition]), flush=True)
            
            datasetIO.save_datamatrix('{0}/{1}_intermediate_embedding_layer{2!s}_finetuning{3!s}_step{4!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), embed[partition])
            datasetIO.save_datamatrix('{0}/{1}_intermediate_embedding_layer{2!s}_finetuning{3!s}_step{4!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), embed[partition])
            
            if d['current_apply_activation_to_embedding']:
                datasetIO.save_datamatrix('{0}/{1}_intermediate_embedding_preactivation_layer{2!s}_finetuning{3!s}_step{4!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), embed_preactivation[partition])
                datasetIO.save_datamatrix('{0}/{1}_intermediate_embedding_preactivation_layer{2!s}_finetuning{3!s}_step{4!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), embed_preactivation[partition])


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
        fg.savefig('{0}/intermediate_reconstructions_layer{1!s}_finetuning{2!s}_step{3!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), transparent=True, pad_inches=0, dpi=1200)
        plt.close()


        # plot 2d embedding
        if d['current_dimensions'][-1] == 2:
            print('plotting 2d embedding...', flush=True)
            fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
            ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
            ax.plot(embed['train'].matrix[:,0], embed['train'].matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.plot(embed['valid'].matrix[:,0], embed['valid'].matrix[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
            ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False, pad=4)
            ax.set_frame_on(False)
            fg.savefig('{0}/intermediate_embedding_layer{1!s}_finetuning{2!s}_step{3!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), transparent=True, pad_inches=0, dpi=600)
            plt.close()
            
            if d['current_apply_activation_to_embedding']:
                fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
                ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
                ax.plot(embed_preactivation['train'].matrix[:,0], embed_preactivation['train'].matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
                ax.plot(embed_preactivation['valid'].matrix[:,0], embed_preactivation['valid'].matrix[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
                ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False, pad=4)
                ax.set_frame_on(False)
                fg.savefig('{0}/intermediate_embedding_preactivation_layer{1!s}_finetuning{2!s}_step{3!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), transparent=True, pad_inches=0, dpi=600)
                plt.close()
        # plot heatmap
        else:
            print('plotting embedding heatmap...', flush=True)
            for partition in partitions:
                if 'all' not in embed:
                    embed['all'] = copy.deepcopy(embed[partition])
                else:
                    embed['all'].append(embed[partition], 0)
            embed['all'].cluster('all', 'cosine', 'average')
            embed['all'].heatmap(rowmetalabels=[], columnmetalabels=[], normalize=False, standardize=False, normalizebeforestandardize=True, cmap_name='bwr', ub=None, lb=None, savefilename='{0}/intermediate_embedding_heatmap_layer{1!s}_finetuning{2!s}_step{3!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), closefigure=True, dpi=300)
            if d['current_apply_activation_to_embedding']:
                for partition in partitions:
                    if 'all' not in embed_preactivation:
                        embed_preactivation['all'] = copy.deepcopy(embed_preactivation[partition])
                    else:
                        embed_preactivation['all'].append(embed_preactivation[partition], 0)
                embed_preactivation['all'].cluster('all', 'cosine', 'average')
                embed_preactivation['all'].heatmap(rowmetalabels=[], columnmetalabels=[], normalize=False, standardize=False, normalizebeforestandardize=True, cmap_name='bwr', ub=None, lb=None, savefilename='{0}/intermediate_embedding_preactivation_heatmap_layer{1!s}_finetuning{2!s}_step{3!s}.png'.format(d['output_path'], d['current_hidden_layer'], d['current_finetuning_run'], selected_step), closefigure=True, dpi=300)
                

    print('done get_sdae_features.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize SDAE features at selected steps for one or more models.')
    parser.add_argument('visualizations_path', help='path to .txt file containing design paths and selected step for models to be visualized', type=str)
    args = parser.parse_args()
    main(args.visualizations_path)



'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import datasetIO
import dataclasses
import copy
import sys


def relu(mat):
    mat[mat < 0] = 0
    return mat

def sdae_transform(dm, W, Be, activation, apply_activation_to_output=False):
    mat = dm.matrix.copy()
    for i, (w, b) in enumerate(zip(W, Be)):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w) + b)
        else:
            mat = mat.dot(w) + b
    em = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=copy.deepcopy(dm.rowmeta),
                                columnname='latent_component',
                                columnlabels=np.array(['LC'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                columnmeta={'activation_applied':np.full(mat.shape[1], apply_activation_to_output, dtype='bool')},
                                matrixname='sdae_transform_of_'+dm.matrixname,
                                matrix=mat)
    return em

def sdae_inverse_transform(em, W, Bd, activation, apply_activation_to_output=False):
    if ~(em.columnmeta['activation_applied'].any()):
        mat = activation(em.matrix)
    else:
        mat = em.matrix.copy()
    for i, (w, b) in enumerate(zip(W[::-1], Bd[::-1])):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w.T) + b)
        else:
            mat = mat.dot(w.T) + b
    rm = dataclasses.datamatrix(rowname=em.rowname,
                                rowlabels=em.rowlabels.copy(),
                                rowmeta=copy.deepcopy(em.rowmeta),
                                columnname='reconstructed_feature',
                                columnlabels=np.array(['RF'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                columnmeta={},
                                matrixname='reconstruction_from_'+em.matrixname,
                                matrix=mat)
    return rm

def sdae_reconstruction(dm, W, Be, Bd, activation, apply_activation_to_output=False, return_embedding=False, return_reconstruction_error=False):
    mat = dm.matrix.copy()
    for i, (w, b) in enumerate(zip(W, Be)):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w) + b)
        else:
            mat = mat.dot(w) + b
    if return_embedding:
        em = dataclasses.datamatrix(rowname=dm.rowname,
                                    rowlabels=dm.rowlabels.copy(),
                                    rowmeta=copy.deepcopy(dm.rowmeta),
                                    columnname='latent_component',
                                    columnlabels=np.array(['LC'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                    columnmeta={'activation_applied':np.full(mat.shape[1], apply_activation_to_output, dtype='bool')},
                                    matrixname='sdae_transform_of_'+dm.matrixname,
                                    matrix=mat.copy())
    if not apply_activation_to_output:
        mat = activation(mat)
    for i, (w, b) in enumerate(zip(W[::-1], Bd[::-1])):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w.T) + b)
        else:
            mat = mat.dot(w.T) + b
    rm = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=copy.deepcopy(dm.rowmeta),
                                columnname='reconstructed_' + dm.columnname,
                                columnlabels=dm.columnlabels.copy(),
                                columnmeta=copy.deepcopy(dm.columnmeta),
                                matrixname='reconstruction_from_sdae_transform_of_'+dm.matrixname,
                                matrix=mat)
    reconstruction_error = np.mean((rm.matrix - dm.matrix)**2)
    if return_embedding and return_reconstruction_error:
        return rm, em, reconstruction_error
    elif return_embedding:
        return rm, em
    elif return_reconstruction_error:
        return rm, reconstruction_error
    else:
        return rm


def main(study_name='your_study'):
    
    # load the data
    orientation = 'fat'
    partitions = ['train', 'valid', 'test']
    
    dataset = {}
    for partition in partitions:
        dataset[partition] = datasetIO.load_datamatrix('data/prepared_data/{0}/{1}.pickle'.format(orientation, partition))
        if 'all' not in dataset:
            dataset['all'] = copy.deepcopy(dataset[partition])
        else:
            dataset['all'].append(dataset[partition], 0)
    
    dataset[study_name] = {}
    for partition in partitions:
        dataset[study_name][partition] = datasetIO.load_datamatrix('data/prepared_data/{0}/{1}/{2}.pickle'.format(study_name, orientation, partition))
        if 'all' not in dataset[study_name]:
            dataset[study_name]['all'] = copy.deepcopy(dataset[study_name][partition])
        else:
            dataset[study_name]['all'].append(dataset[study_name][partition], 0)
    
    partitions.append('all')
    
    
    # create output directories
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/sdae_features'):
        os.mkdir('results/sdae_features')
    if not os.path.exists('results/sdae_features/{0}'.format(study_name)):
        os.mkdir('results/sdae_features/{0}'.format(study_name))
    if not os.path.exists('results/sdae_features/{0}/{1}'.format(study_name, orientation)):
        os.mkdir('results/sdae_features/{0}/{1}'.format(study_name, orientation))
    
    
    # load the model
    activation_function, activation_function_name = (relu, 'relu')
    with open('results/autoencoder/fat/ns5_last2_first0.05_5layers_relu_variables.pickle', 'rb') as fr:
        W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
    
    
    # get embeddings and reconstructions
    sdae = {}
    for partition in partitions:
        sdae[partition] = {}
        sdae[partition]['recon'], sdae[partition]['embed'], sdae[partition]['error'] = sdae_reconstruction(dataset[partition], W, Be, Bd, activation=activation_function, apply_activation_to_output=False, return_embedding=True, return_reconstruction_error=True)
        print('{0} error: {1:1.3g}'.format(partition, sdae[partition]['error']))
    
    sdae[study_name] = {}
    for partition in partitions:
        sdae[study_name][partition] = {}
        sdae[study_name][partition]['recon'], sdae[study_name][partition]['embed'], sdae[study_name][partition]['error'] = sdae_reconstruction(dataset[study_name][partition], W, Be, Bd, activation=activation_function, apply_activation_to_output=False, return_embedding=True, return_reconstruction_error=True)
        print('{0} {1} error: {2:1.3g}'.format(study_name, partition, sdae[study_name][partition]['error']))
    
    
    # visualize embedding
    if sdae['all']['embed'].shape[1] < 5:
        for nx in range(sdae['all']['embed'].shape[1]-1):
            for ny in range(nx+1, sdae['all']['embed'].shape[1]):
                
                #tissues = np.unique(dataset['all'].rowmeta['general_tissue'])
                tissues = ['Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel', 'Brain',
                           'Breast', 'Colon', 'Esophagus', 'Heart', 'Kidney', 'Liver', 'Lung', 'Muscle',
                           'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Salivary Gland', 'Skin',
                           'Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Uterus', 'Vagina']
                tissue_abbrevs = ['AT', 'AG', 'B', 'BV', 'Bn',
                                  'Bt', 'C', 'E', 'H', 'K', 'Lr', 'Lg', 'M',
                                  'N', 'O', 'Ps', 'Py', 'Pe', 'SG', 'Sk',
                                  'SI', 'Sp', 'St', 'Ts', 'Td', 'U', 'V']
                cmap = plt.get_cmap('gist_rainbow')
                colors = [cmap(float((i+0.5)/len(tissues))) for i in range(len(tissues))]
                
                fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                for tissue, tissue_abbrev, color in zip(tissues, tissue_abbrevs, colors):
                    if tissue == '-666':
                        continue
                #        zorder = 0
                #        alpha = 0.05
                #        color = 'k'
                    else:
                        zorder = 1
                        alpha = 0.5
                    hit = dataset['all'].rowmeta['general_tissue'] == tissue
                    hidxs = hit.nonzero()[0]
                #    ax.plot(sdae['all']['embed'].matrix[hit,nx], sdae['all']['embed'].matrix[hit,ny], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                    ax.plot(sdae['all']['embed'].matrix[hit,nx], sdae['all']['embed'].matrix[hit,ny], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=0.2, markeredgewidth=0, alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                    for hidx in hidxs:
                        ax.text(sdae['all']['embed'].matrix[hidx,nx], sdae['all']['embed'].matrix[hidx,ny], tissue_abbrev, horizontalalignment='center', verticalalignment='center', fontsize=4, color=color, alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                ax.plot(sdae[study_name]['all']['embed'].matrix[:,nx], sdae[study_name]['all']['embed'].matrix[:,ny], linestyle='None', linewidth=0, marker='x', markerfacecolor='k', markeredgecolor='k', markersize=0.2, markeredgewidth=0, alpha=1, zorder=1, label=study_name)
                for hidx in range(sdae[study_name]['all']['embed'].shape[0]):
                    ax.text(sdae[study_name]['all']['embed'].matrix[hidx,nx], sdae[study_name]['all']['embed'].matrix[hidx,ny], 'X', horizontalalignment='center', verticalalignment='center', fontsize=4, color='k', alpha=1, zorder=1, label=study_name)
                ax.set_xlim(sdae['all']['embed'].matrix[:,nx].min(), sdae['all']['embed'].matrix[:,nx].max())
                ax.set_ylim(sdae['all']['embed'].matrix[:,ny].min(), sdae['all']['embed'].matrix[:,ny].max())
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=40, fontsize=8, labelspacing=0.25)
                ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
                ax.set_frame_on(False)
                fg.savefig('results/sdae_features/{0}/{1}/sdae2d_{2}_coloredby_general_tissue_x{3!s}_y{4!s}.png'.format(study_name, orientation, activation_function_name, nx, ny), transparent=True, pad_inches=0, dpi=600)
                ax.set_xlim(sdae[study_name]['all']['embed'].matrix[:,nx].min(), sdae[study_name]['all']['embed'].matrix[:,nx].max())
                ax.set_ylim(sdae[study_name]['all']['embed'].matrix[:,ny].min(), sdae[study_name]['all']['embed'].matrix[:,ny].max())
                fg.savefig('results/sdae_features/{0}/{1}/sdae2d_{2}_coloredby_general_tissue_x{3!s}_y{4!s}_zoom.png'.format(study_name, orientation, activation_function_name, nx, ny), transparent=True, pad_inches=0, dpi=600)
                plt.close()
    
    
    # save embedding
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae2d_{2}_datamatrix.txt.gz'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['embed'])
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae2d_{2}_datamatrix.pickle'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['embed'])
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae_reconstructions_{2}_datamatrix.txt.gz'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['recon'])
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae_reconstructions_{2}_datamatrix.pickle'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['recon'])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError('too many inputs')
'''
