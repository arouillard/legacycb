# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import sys
sys.path.append('../../utilities')

import argparse
import numpy as np
import pandas as pd
import dataclasses as dc
import datasetIO
import os
import copy
from matplotlib import pyplot as plt


def main(model_folders_path):
    
    print('reading list of model folders...', flush=True)
    with open(model_folders_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        model_folders = fr.read().split('\n')
#    if '_v' in model_folders_path:
#        version = model_folders_path.replace('.txt', '').split('_')[-1]
    
    print('loading input datamatrix...', flush=True)
    model_folder_parts = model_folders[0].split('/')
    dataset_name = model_folder_parts[model_folder_parts.index('hp_search')+1]
    observed_ = datasetIO.load_datamatrix('../../input_data/{0}/datamatrix.pickle'.format(dataset_name))
    print(observed_, flush=True)
    
    print('attaching hla types...', flush=True)
    columnlabel_idx = {l:i for i,l in enumerate(observed_.columnlabels)}
    hla_types_df = pd.read_csv('../../original_data/1000genomes/20140702_hla_diversity.csv', index_col=False)
    for metalabel in hla_types_df.columns.values[1:]:
        observed_.columnmeta[metalabel] = np.full(observed_.shape[1], 'NA', dtype='object')
        for columnlabel, value in zip(hla_types_df['id'].values, hla_types_df[metalabel].values):
            if columnlabel in columnlabel_idx:
                columnidx = columnlabel_idx[columnlabel]
                observed_.columnmeta[metalabel][columnidx] = value
        uvals, counts = np.unique(observed_.columnmeta[metalabel], return_counts=True)
        max_num_uvals = 25
        if uvals.size > max_num_uvals:
            si = np.argsort(counts)[::-1]
            low_freq_uvals = uvals[si[max_num_uvals:]]
            observed_.columnmeta[metalabel][np.in1d(observed_.columnmeta[metalabel], low_freq_uvals)] = 'NA'
    
    for model_folder in model_folders:
        
        print('working on model_folder: {0}...'.format(model_folder), flush=True)
        input_path = '{0}/embedding.csv.gz'.format(model_folder)
        output_folder = '/'.join(model_folder.replace('/hp_search/', '/output_data/').split('/')[:-1]) + '/embeddings'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path_prefix = '{0}/{1}'.format(output_folder, model_folder.split('/')[-1])
        print('input_path: {0}'.format(input_path), flush=True)
        print('output_folder: {0}'.format(output_folder), flush=True)
        print('output_path_prefix: {0}'.format(output_path_prefix), flush=True)
        
        if os.path.exists(input_path):
            
            print('loading embedding datamatrix...', flush=True)
            df = pd.read_csv(input_path, index_col=False, usecols=[observed_.rowname, 'Latent1', 'Latent2'])
            hidden = dc.datamatrix(rowname=observed_.rowname,
                                   rowlabels=df[observed_.rowname].values,
                                   rowmeta={},
                                   columnname='latent_component',
                                   columnlabels=np.array(['Latent1', 'Latent2'], dtype='object'),
                                   columnmeta={},
                                   matrixname=observed_.rowname + '_embedding_from_' + observed_.matrixname,
                                   matrix=np.concatenate((df.Latent1.values.reshape(-1,1), df.Latent2.values.reshape(-1,1)), 1))
            del df
            print(hidden, flush=True)
            
            print('aligning input datamatrix and embedding datamatrix...', flush=True)
            if observed_.shape[0] == hidden.shape[0] and (observed_.rowlabels == hidden.rowlabels).all():
                observed = copy.deepcopy(observed_)
            else:
                observed = observed_.tolabels(rowlabels=hidden.rowlabels.copy())
            hidden.rowmeta = copy.deepcopy(observed.rowmeta)
            print(observed, flush=True)
            
            # visualization
            print('plotting embedding...', flush=True)
            fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
            ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
            ax.plot(hidden.matrix[:,0], hidden.matrix[:,1], 'ok', markersize=1, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
#            ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
            ax.set_frame_on(False)
            fg.savefig('{0}.png'.format(output_path_prefix), transparent=True, pad_inches=0, dpi=300)
            plt.close()
            for metalabel in ['mean', 'stdv', 'position']:
                z = hidden.rowmeta[metalabel].astype('float64')
                fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                ax.scatter(hidden.matrix[:,0], hidden.matrix[:,1],  s=1, c=z, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), alpha=0.5, vmin=z.min(), vmax=z.max())
                ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
#                ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
                ax.set_frame_on(False)
                fg.savefig('{0}_colored_by_{1}.png'.format(output_path_prefix, metalabel), transparent=True, pad_inches=0, dpi=300)
                plt.close()
            for metalabel in ['gene_name']:
                categories = np.unique(hidden.rowmeta[metalabel])
                cmap = plt.get_cmap('gist_rainbow')
                colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
                fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                for category, color in zip(categories, colors):
                    if category == 'NA':
                        color = 'k'
                        alpha = 0.1
                        zorder = 0
                    else:
                        alpha = 0.5
                        zorder = 1
                    hit = hidden.rowmeta[metalabel] == category
                    ax.plot(hidden.matrix[hit,0], hidden.matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=category)
                ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
#                ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
                ax.set_frame_on(False)
                fg.savefig('{0}_colored_by_{1}.png'.format(output_path_prefix, metalabel), transparent=True, pad_inches=0, dpi=300)
                plt.close()
            hla_hit = np.array(['HLA-' in x for x in hidden.rowmeta['gene_name']], dtype='bool')
            hla_names = hidden.rowmeta['gene_name'].copy()
            hla_names[~hla_hit] = 'NA'
            categories = np.unique(hla_names)
            cmap = plt.get_cmap('gist_rainbow')
            colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
            fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
            ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
            for category, color in zip(categories, colors):
                if category == 'NA':
                    color = 'k'
                    alpha = 0.1
                    zorder = 0
                else:
                    alpha = 0.5
                    zorder = 1
                hit = hla_names == category
                ax.plot(hidden.matrix[hit,0], hidden.matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=1, markeredgewidth=0, alpha=alpha, zorder=zorder, label=category)
            ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
#            ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
            ax.set_frame_on(False)
            fg.savefig('{0}_colored_by_hlagene.png'.format(output_path_prefix), transparent=True, pad_inches=0, dpi=300)
            plt.close()
            
            print('computing right factor matrix...', flush=True)
            rightfactormat, residuals, rank, singular_values = np.linalg.lstsq(hidden.matrix, observed.matrix)
            factored = dc.datamatrix(rowname=observed.columnname,
                                     rowlabels=observed.columnlabels.copy(),
                                     rowmeta=copy.deepcopy(observed.columnmeta),
                                     columnname='latent_component',
                                     columnlabels=np.array(['Latent1', 'Latent2'], dtype='object'),
                                     columnmeta={},
                                     matrixname=observed.columnname + '_embedding_from_' + observed.matrixname,
                                     matrix=rightfactormat.T)
            
            print('plotting transpose embedding...', flush=True)
            fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
            ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
            ax.plot(factored.matrix[:,0], factored.matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
            ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
#            ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
            ax.set_frame_on(False)
            fg.savefig('{0}_transpose.png'.format(output_path_prefix), transparent=True, pad_inches=0, dpi=300)
            plt.close()
            for metalabel in factored.rowmeta: # ['population', 'super_population', 'gender']:
                categories = np.unique(factored.rowmeta[metalabel])
                cmap = plt.get_cmap('gist_rainbow')
                colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
                fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                for category, color in zip(categories, colors):
                    if category == 'NA':
                        color = 'k'
                        alpha = 0.1
                        zorder = 0
                    else:
                        alpha = 0.5
                        zorder = 1
                    hit = factored.rowmeta[metalabel] == category
                    ax.plot(factored.matrix[hit,0], factored.matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=category)
                ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
#                ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
                ax.set_frame_on(False)
                fg.savefig('{0}_transpose_colored_by_{1}.png'.format(output_path_prefix, metalabel), transparent=True, pad_inches=0, dpi=300)
                plt.close()

    print('done plot_embeddings.py', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create figures from one or more models.')
    parser.add_argument('model_folders_path', help='path to .txt file containing model folders', type=str)
    args = parser.parse_args()
    main(args.model_folders_path)
