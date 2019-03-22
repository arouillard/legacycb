# -*- coding: utf-8 -*-
"""
Clinical Outcome Classifier
get scores for individual features
multiple hypothesis correction by dataset or altogether?
what is the best test?
"""

import sys
#custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import os
#import gzip
import numpy as np
import copy
from machinelearning import datasetselection, featureselection
import machinelearning.dataclasses as dc
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import fastcluster

# load distance matrix
with open('gene_gene_matrix_euclidean_distance_from_projection.pickle', 'rb') as fr:
    gene_gene = pickle.load(fr)

# prefer ward linkage for euclidean distance or at least this case
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'ward')
si = hierarchy.leaves_list(lnk).astype('int64')

# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)
if ~(gene_proj.rowlabels == gene_gene.rowlabels).all():
    raise ValueError('genes not aligned')
gene_proj.reorder(si, 0)
ordered_genes = gene_proj.rowlabels.copy()
del gene_gene, lnk, si

# select datasets
dataset_info = datasetselection.finddatasets(getalllevels=True)
included_datasetabbrevs = {'clinvar', 'dbgap_cleaned', 'gad', 'gadhighlevel_cleaned', 'gobp', 'gocc', 'gomf', 'gwascatalog_cleaned', 'gwasdbdisease_cleaned', 'gwasdbphenotype_cleaned', 'hpo', 'hugenavigator', 'humancyc', 'kegg', 'locate', 'locatepredicted', 'mgimpo', 'omim', 'panther', 'reactome', 'wikipathways'}
excluded_datasetabbrevs = set(dataset_info.keys()).difference(included_datasetabbrevs)
for datasetabbrev in excluded_datasetabbrevs:
    del dataset_info[datasetabbrev]

# parameters
feature_selection_test_name = 'permutation'
analysis_version = 'v7chisq14network'
results_layers = {'test_statistic_values', 'pvalues', 'correlation_sign', 'is_significant'}

# iterate over datasets
for datasetabbrev, datasetinfo in dataset_info.items():
    # just work with pathways for testing/debugging the pipeline
#    if 'kegg' not in datasetabbrev: # datasetabbrev not in {'kegg', 'panther', 'reactome', 'wikipathways'}:
#        continue
    if not os.path.exists('univariate_feature_importance_{0}/{1}_pvalues.pickle'.format(analysis_version, datasetabbrev)):
        continue
    print('working on {0}...'.format(datasetabbrev))
    # load aligned matrices
#    print('loading aligned matrices...')
#    with open('aligned_matrices_{0}/gene_atb_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
#        gene_atb = pickle.load(fr)
#    background_size = gene_atb.shape[0]
#    del gene_atb
#    with open('aligned_matrices_{0}/gene_cst_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
#        gene_cst = pickle.load(fr)
    # load pvalues, test statistic values, etc.
    print('loading results...')
    atb_cst = {}
    for layer in results_layers:
        with open('univariate_feature_importance_{0}/{1}_{2}.pickle'.format(analysis_version, datasetabbrev, layer), mode='rb') as fr:
            atb_cst[layer] = pickle.load(fr)
    minp = atb_cst['pvalues'].matrix[atb_cst['pvalues'].matrix > 0].min()
    atb_cst['pvalues'].matrix[atb_cst['pvalues'].matrix == 0] = minp
    # keep only significant features (include effect size cutoff)
    atb_cst['is_significant'].matrix = np.logical_and(atb_cst['is_significant'].matrix, atb_cst['test_statistic_values'].matrix > 0.01)
    tobediscarded = ~(atb_cst['is_significant'].matrix.any(1))
    hit = np.array(['infect' in x or 'microb' in x or 'vir' in x or 'bact' in x or 'biot' in x or 'gut' in x or 'flora' in x for x in atb_cst['pvalues'].rowlabels], dtype='bool')
    tobediscarded = np.logical_or(tobediscarded, ~hit)    
    if tobediscarded.all():
        print('no significant ID terms')
        continue
    for layer in results_layers:
        atb_cst[layer].discard(tobediscarded, 0)
    # order columns (genes) by projection clustering and rows (features) by correlation of signed negative log10 pvalue maps
    atb_cst['snlp'] = copy.deepcopy(atb_cst['pvalues'])
    atb_cst['snlp'].matrix = -1.0*atb_cst['correlation_sign'].matrix*np.log10(atb_cst['pvalues'].matrix)
    gene_idx = {g:i for i,g in enumerate(atb_cst['snlp'].columnlabels)}
    si = np.array([gene_idx[g] for g in ordered_genes if g in gene_idx], dtype='int64')
    atb_cst['snlp'].reorder(si, 1)
    atb_cst['snlp'].cluster(0, metric='correlation', method='average')
    for layer in results_layers:
        atb_cst[layer] = atb_cst[layer].tolabels(rowlabels=atb_cst['snlp'].rowlabels.copy(), columnlabels=atb_cst['snlp'].columnlabels.copy())
    nmax = 100
    with open('univariate_feature_importance_{0}/{1}_IDterms_rankedgenes.txt'.format(analysis_version, datasetabbrev), mode='wt') as fw:
        for term, scores in zip(atb_cst['snlp'].rowlabels, atb_cst['snlp'].matrix):
            si = np.argsort(scores)[::-1][:nmax]
            fw.write('\t'.join([term] + ['{0},{1:1.3g}'.format(atb_cst['snlp'].columnlabels[j], scores[j]) for j in si]) + '\n')
    # plot results
    print('plotting results...')
#    atb_cst['snlp'].heatmap([], [], normalize='rows', savefilename='univariate_feature_importance_{0}/{1}_atb_cst_snlp_heatmap.png'.format(analysis_version, datasetabbrev), closefigure=True, dpi=1200)
#    atb_cst['test_statistic_values'].heatmap([], [], savefilename='univariate_feature_importance_{0}/{1}_atb_cst_test_statistic_values_heatmap.png'.format(analysis_version, datasetabbrev), closefigure=True, dpi=1200)
    gp_in = gene_proj.tolabels(rowlabels=atb_cst['pvalues'].columnlabels.copy())
    gp_out = gene_proj.tolabels(rowlabels=gene_proj.rowlabels[~np.in1d(gene_proj.rowlabels, gp_in.rowlabels)])
    num_sig_atbs = atb_cst['is_significant'].matrix.any(1).sum()
#    num_fig_rows = int(np.ceil(float(num_sig_atbs)/2.0))
    rows = 4
    cols = 8
    yub = 0.5
    ylb = -0.35
    xub = 0.45
    xlb = -0.4
    fg, axs = plt.subplots(rows, cols, figsize=(10,5))
    fg.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.9, wspace=0.2, hspace=0.2)
    fg.suptitle('{0}, {1!s} sig features'.format(datasetinfo['name'], num_sig_atbs), fontsize=8)
    axs = axs.reshape(-1)
    num_axs = rows*cols
    ia = 0
    for feature, tvalues, pvalues, snlps, signs, significances in zip(atb_cst['pvalues'].rowlabels, atb_cst['test_statistic_values'].matrix, atb_cst['pvalues'].matrix, atb_cst['snlp'].matrix, atb_cst['correlation_sign'].matrix, atb_cst['is_significant'].matrix):
        if ia >= num_axs:
            break
        if significances.any():
            ax = axs[ia]
#            cub = np.max(np.abs(snlps))
#            clb = -cub
            cub = snlps.max()
            clb = snlps.min()
            ax.scatter(gp_in.matrix[:,0], gp_in.matrix[:,1], s=0.1, c=snlps, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), vmin=clb, vmax=cub)
            ax.scatter(gp_out.matrix[:,0], gp_out.matrix[:,1], s=0.1, c='k', alpha=0.01, marker='o', edgecolors='none')
            ax.set_title(feature[:35], fontsize=4)
            ax.set_ylim(ylb, yub)
            ax.set_xlim(xlb, xub)
            ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
            ax.set_frame_on(False)
            ax.axvline(xlb, linewidth=0.5, color='k')
            ax.axvline(xub, linewidth=0.5, color='k')
            ax.axhline(ylb, linewidth=0.5, color='k')
            ax.axhline(yub, linewidth=0.5, color='k')
            ia += 1
    if ia < num_axs:
        for j in range(ia, num_axs):
            plt.delaxes(axs[j])
    fg.savefig('univariate_feature_importance_{0}/{1}_maps_IDterms.png'.format(analysis_version, datasetabbrev), transparent=True, pad_inches=0, dpi=300)
#    fg.show()
    plt.close()
    
print('done.')

#from machinelearning import datasetIO
#for layer in ['test_statistic_values', 'pvalues', 'snlp', 'correlation_sign', 'is_significant']:
#    datasetIO.save_datamatrix('univariate_feature_importance_{0}/{1}_gene_atb_{2}.txt'.format(analysis_version, datasetabbrev, layer), atb_cst[layer].totranspose())
