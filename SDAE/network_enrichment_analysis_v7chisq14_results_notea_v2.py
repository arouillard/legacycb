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
import machinelearning.stats as mlstats
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import fastcluster

# load distance matrix
with open('gene_gene_matrix_euclidean_distance_from_projection.pickle', 'rb') as fr:
    gene_gene = pickle.load(fr)
'''
si = {}
cst = {}
numcst = 87
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'average')
plt.figure(); hierarchy.dendrogram(lnk)
si['average'] = hierarchy.leaves_list(lnk)
cst['average'] = hierarchy.cut_tree(lnk, n_clusters=numcst).reshape(-1)
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'centroid')
plt.figure(); hierarchy.dendrogram(lnk)
si['centroid'] = hierarchy.leaves_list(lnk)
cst['centroid'] = hierarchy.cut_tree(lnk, n_clusters=285).reshape(-1) # numcst).reshape(-1) # does not return the correct number of clusters when for centroid linkage
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'ward')
plt.figure(); hierarchy.dendrogram(lnk)
si['ward'] = hierarchy.leaves_list(lnk)
cst['ward'] = hierarchy.cut_tree(lnk, n_clusters=numcst).reshape(-1)
# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)
if ~(gene_proj.rowlabels == gene_gene.rowlabels).all():
    raise ValueError('genes not aligned')
# evaluate the linkage methods
# ward appears to be best, average pretty good, centroid terrible (but may be because something is wrong with the code)
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/numcst)) for i in range(numcst)]
for linkage_method, cluster_ids in cst.items():
    fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
    ax.set_position([0.15/6.5, 0.05/5.3, 5.0/6.5, 4.9/5.3])
    for cluster_id, color in enumerate(colors):
        hit = cluster_ids == cluster_id
        ax.plot(gene_proj.matrix[hit,0], gene_proj.matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=0.5, zorder=1, label=str(cluster_id))
    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    ax.set_title(linkage_method, fontsize=8)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, ncol=3, columnspacing=0, markerscale=4, fontsize=8, labelspacing=0.4, handletextpad=0)
    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
    ax.set_frame_on(False)
    fg.savefig('gene_atb_matrix_2d_dnn_projection_{0!s}clusters_{1}linkage.png'.format(numcst, linkage_method), transparent=True, pad_inches=0, dpi=600)
    fg.show()
'''
# prefer ward linkage for euclidean distance or at least this case
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'ward')
#plt.figure()
#hierarchy.dendrogram(lnk)
si = hierarchy.leaves_list(lnk).astype('int64')
#cst = hierarchy.cut_tree(lnk, n_clusters=87).reshape(-1)

# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)
if ~(gene_proj.rowlabels == gene_gene.rowlabels).all():
    raise ValueError('genes not aligned')
gene_proj.reorder(si, 0)
#gene_gene.reorder(si, 0)
#gene_gene.reorder(si, 1)
ordered_genes = gene_proj.rowlabels.copy()
del gene_gene, lnk, si

# load assocations between features and tissues from ordinary enrichment analysis
with open('dataset_feature_tissue_dict.pickle', 'rb') as fr:
    dataset_feature_tissue = pickle.load(fr)
with open('dataset_feature_tissuespecific_dict.pickle', 'rb') as fr:
    dataset_feature_tissuespecific = pickle.load(fr)

# select datasets
dataset_info = datasetselection.finddatasets(getalllevels=True)
included_datasetabbrevs = {'clinvar', 'dbgap_cleaned', 'gad', 'gadhighlevel_cleaned', 'gobp', 'gocc', 'gomf', 'gwascatalog_cleaned', 'gwasdbdisease_cleaned', 'gwasdbphenotype_cleaned', 'hpo', 'hugenavigator', 'humancyc', 'kegg', 'locate', 'locatepredicted', 'mgimpo', 'omim', 'panther', 'reactome', 'wikipathways'}
excluded_datasetabbrevs = set(dataset_info.keys()).difference(included_datasetabbrevs)
for datasetabbrev in excluded_datasetabbrevs:
    del dataset_info[datasetabbrev]
dataset_info['gtex_overexpression_nonspecific'] = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_nonspecific', 'name':'Tissue Over-expression 10-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_nonspecific.pickle'}
dataset_info['gtex_overexpression_toptissue'] = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_toptissue', 'name':'Top Tissue Over-expression 10-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_toptissue.pickle'}
dataset_info['gtex_overexpression_specific'] = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_specific', 'name':'Specific Tissue Over-expression 10-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_3xspecific.pickle'}
    
# parameters
feature_selection_test_name = 'permutation'
analysis_version = 'v7chisq14network'
results_layers = {'test_statistic_values', 'pvalues', 'correlation_sign', 'is_significant'}

# load significant tissue snlp profiles
with open('univariate_feature_importance_{0}/{1}_{2}_significant.pickle'.format(analysis_version, 'gtex_overexpression_nonspecific', 'snlp'), mode='rb') as fr:
    tissue_cst_nonspecific = pickle.load(fr)
tissue_cst_nonspecific.discard(tissue_cst_nonspecific.rowlabels=='none', 0)
with open('univariate_feature_importance_{0}/{1}_{2}_significant.pickle'.format(analysis_version, 'gtex_overexpression_specific', 'snlp'), mode='rb') as fr:
    tissue_cst_specific = pickle.load(fr)
tissue_cst_specific.discard(tissue_cst_specific.rowlabels=='none', 0)
    
# iterate over datasets
for datasetabbrev, datasetinfo in dataset_info.items():
    # just work with pathways for testing/debugging the pipeline
    if 'gtex_overexpression' in datasetabbrev: # datasetabbrev not in {'kegg', 'panther', 'reactome', 'wikipathways'}:
        continue
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
#    plt.figure()
#    plt.plot(atb_cst['test_statistic_values'].matrix.reshape(-1), -np.log10(atb_cst['pvalues'].matrix.reshape(-1)), '.k')
#    plt.title(datasetabbrev)
#    plt.show()
#    plt.figure()
#    plt.hist(atb_cst['test_statistic_values'].matrix.reshape(-1), bins=20, range=(0,0.2))
#    plt.show()
    # keep only significant features (include effect size cutoff) and features not discovered by ordinary tissue enrichment analysis
    atb_cst['is_significant'].matrix = np.logical_and(atb_cst['is_significant'].matrix, atb_cst['test_statistic_values'].matrix > 0.1)
    tobediscarded = ~(atb_cst['is_significant'].matrix.any(1))
    if datasetabbrev in dataset_feature_tissue:
        tobediscarded = np.logical_or(tobediscarded, np.in1d(atb_cst['is_significant'].rowlabels, list(dataset_feature_tissue[datasetabbrev].keys())))
    if datasetabbrev in dataset_feature_tissuespecific:
        tobediscarded = np.logical_or(tobediscarded, np.in1d(atb_cst['is_significant'].rowlabels, list(dataset_feature_tissuespecific[datasetabbrev].keys())))
    # also discard features with snlp maps correlated with tissues
    atb_cst_mat = -1.0*atb_cst['correlation_sign'].matrix*np.log10(atb_cst['pvalues'].matrix)
    commongenes = atb_cst['pvalues'].columnlabels[np.in1d(atb_cst['pvalues'].columnlabels, tissue_cst_nonspecific.columnlabels)]
    gene_idx = {g:i for i,g in enumerate(atb_cst['pvalues'].columnlabels)}
    si = np.array([gene_idx[g] for g in commongenes], dtype='int64')
    atb_cst_mat = atb_cst_mat[:,si]
    gene_idx = {g:i for i,g in enumerate(tissue_cst_nonspecific.columnlabels)}
    si = np.array([gene_idx[g] for g in commongenes], dtype='int64')
    tis_cst_mat = tissue_cst_nonspecific.matrix[:,si]
    atb_tis_rvalues, atb_tis_pvalues = mlstats.corr(atb_cst_mat, tis_cst_mat, axis=0, metric='pearson', getpvalues=True)
    gene_idx = {g:i for i,g in enumerate(tissue_cst_specific.columnlabels)}
    si = np.array([gene_idx[g] for g in commongenes], dtype='int64')
    tis_cst_mat = tissue_cst_specific.matrix[:,si]
    atb_tis_specific_rvalues, atb_tis_specific_pvalues = mlstats.corr(atb_cst_mat, tis_cst_mat, axis=0, metric='pearson', getpvalues=True)
    atb_tis_rvalues = np.append(atb_tis_rvalues, atb_tis_specific_rvalues, 1)
    atb_tis_pvalues = np.append(atb_tis_pvalues, atb_tis_specific_pvalues, 1)
    atb_tis_is_sig = featureselection.multiple_hypothesis_testing_correction(atb_tis_pvalues, alpha=0.05, method='fdr_by')[0]
    atb_tis_is_sig = np.logical_and(atb_tis_is_sig, atb_tis_rvalues > 0.1)
    tobediscarded = np.logical_or(tobediscarded, atb_tis_is_sig.any(1))
    if tobediscarded.any():
        print('{0!s} features are enriched in at least one tissue.'.format(tobediscarded.sum() - (~(atb_cst['is_significant'].matrix.any(1))).sum()))
    if tobediscarded.all():
        print('no interesting features remaining')
        continue
    for layer in results_layers:
        atb_cst[layer].discard(tobediscarded, 0)
    # order columns (genes) by projection clustering and rows (features) by correlation of signed negative log10 pvalue maps
    atb_cst['snlp'] = copy.deepcopy(atb_cst['pvalues'])
    atb_cst['snlp'].matrix = -1.0*atb_cst['correlation_sign'].matrix*np.log10(atb_cst['pvalues'].matrix)
    gene_idx = {g:i for i,g in enumerate(atb_cst['snlp'].columnlabels)}
    si = np.array([gene_idx[g] for g in ordered_genes if g in gene_idx], dtype='int64')
    atb_cst['snlp'].reorder(si, 1)
    atb_cst['snlp'].cluster(0, metric='cosine', method='average')
    for layer in results_layers:
        atb_cst[layer] = atb_cst[layer].tolabels(rowlabels=atb_cst['snlp'].rowlabels.copy(), columnlabels=atb_cst['snlp'].columnlabels.copy())
    # plot results
    print('plotting results...')
#    atb_cst['snlp'].heatmap([], [], normalize='rows', savefilename='univariate_feature_importance_{0}/{1}_atb_cst_snlp_heatmap_notea.png'.format(analysis_version, datasetabbrev), closefigure=True, dpi=1200)
#    atb_cst['test_statistic_values'].heatmap([], [], savefilename='univariate_feature_importance_{0}/{1}_atb_cst_test_statistic_values_heatmap_notea.png'.format(analysis_version, datasetabbrev), closefigure=True, dpi=1200)
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
    fg.savefig('univariate_feature_importance_{0}/{1}_maps_notea_v2.png'.format(analysis_version, datasetabbrev), transparent=True, pad_inches=0, dpi=1200)
#    fg.show()
    plt.close()
print('done.')
