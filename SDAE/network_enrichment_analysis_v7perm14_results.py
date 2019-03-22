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

# load clusters
#with open('gene_gene_matrix_euclidean_distance.pickle', 'rb') as fr:
#    gene_cluster = pickle.load(fr)

# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)

# select datasets
dataset_info = datasetselection.finddatasets(getalllevels=True)
included_datasetabbrevs = {'clinvar', 'dbgap_cleaned', 'gad', 'gadhighlevel_cleaned', 'gobp', 'gocc', 'gomf', 'gwascatalog_cleaned', 'gwasdbdisease_cleaned', 'gwasdbphenotype_cleaned', 'hpo', 'hugenavigator', 'humancyc', 'kegg', 'locate', 'locatepredicted', 'mgimpo', 'omim', 'panther', 'reactome', 'wikipathways'}
excluded_datasetabbrevs = set(dataset_info.keys()).difference(included_datasetabbrevs)
for datasetabbrev in excluded_datasetabbrevs:
    del dataset_info[datasetabbrev]
    
# analysis version
analysis_version = 'v7perm14network'
if not os.path.exists('datasets_in_progress_{0}'.format(analysis_version)):
    os.mkdir('datasets_in_progress_{0}'.format(analysis_version))
if not os.path.exists('feature_groups_{0}'.format(analysis_version)):
    os.mkdir('feature_groups_{0}'.format(analysis_version))
if not os.path.exists('univariate_feature_importance_{0}'.format(analysis_version)):
    os.mkdir('univariate_feature_importance_{0}'.format(analysis_version))

# parameters
feature_selection_test_name = 'permutation'

# iterate over datasets
for datasetabbrev, datasetinfo in dataset_info.items():
    # just work with pathways for testing/debugging the pipeline
    if datasetabbrev != 'gadhighlevel_cleaned': # not in {'kegg', 'panther', 'reactome', 'wikipathways'}:
        continue
    print('working on {0}...'.format(datasetabbrev))
    # load aligned matrices
    print('loading aligned matrices...')
    with open('aligned_matrices_{0}/gene_atb_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
        gene_atb = pickle.load(fr)
    background_size = gene_atb.shape[0]
    del gene_atb
#    with open('aligned_matrices_{0}/gene_cst_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
#        gene_cst = pickle.load(fr)
    # load pvalues and test statistic values
    print('getting results...')
    atb_cst = {}
    with open('aligned_matrices_{0}/atb_cst_{1}_pvalues.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
        atb_cst['pvalues'] = pickle.load(fr)
    with open('aligned_matrices_{0}/atb_cst_{1}_test_statistic_values.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
        atb_cst['test_statistic_values'] = pickle.load(fr)
    # compute other values
    atb_cst['pvalues_corrected'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['correlation_sign'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['is_significant'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['correlation_sign'].matrix = np.sign(atb_cst['test_statistic_values'].matrix)
    atb_cst['is_significant'].matrix, atb_cst['pvalues_corrected'].matrix = featureselection.multiple_hypothesis_testing_correction(atb_cst['pvalues'].matrix, alpha=0.05, method='fdr_by')
    # update dtypes
    for layername in atb_cst:
        atb_cst[layername].updatedtypeattribute()
    # save results
    print('writing results...')
#    for layername, layer in atb_cst.items():
#        with open('univariate_feature_importance_{0}/{1}_{2}.pickle'.format(analysis_version, datasetabbrev, layername), mode='wb') as fw:
#            pickle.dump(layer, fw)
    with open('univariate_feature_importance_{0}/{1}.txt'.format(analysis_version, datasetabbrev), mode='wt', encoding="utf-8", errors="surrogateescape") as fw, \
         open('univariate_feature_importance_{0}/{1}_filtered.txt'.format(analysis_version, datasetabbrev), mode='wt', encoding="utf-8", errors="surrogateescape") as fwf:
        writelist = ['cluster',
                     'dataset',
                     'abbreviation',
                     'feature',
                     'test',
                     'test_statistic_values',
                     'pvalue',
                     'pvalue_corrected',
                     'is_significant',
                     'correlation_sign',
                     'feature_size',
#                     'cluster_size',
                     'background_size',
#                     'intersection',
#                     'union',
                     'preferred_correlated_rowstat',
                     'correlated_features']
        fw.write('\t'.join(writelist) + '\n')
        fwf.write('\t'.join(writelist) + '\n')
        for i, feature in enumerate(atb_cst['pvalues'].rowlabels):
            for j, cluster in enumerate(atb_cst['pvalues'].columnlabels):
                writelist = [cluster,
                             datasetinfo['name'],
                             datasetabbrev,
                             feature,
                             feature_selection_test_name,
                             '{0:1.5g}'.format(atb_cst['test_statistic_values'].matrix[i,j]),
                             '{0:1.5g}'.format(atb_cst['pvalues'].matrix[i,j]),
                             '{0:1.5g}'.format(atb_cst['pvalues_corrected'].matrix[i,j]),
                             '{0:1.5g}'.format(atb_cst['is_significant'].matrix[i,j]),
                             '{0:1.5g}'.format(atb_cst['correlation_sign'].matrix[i,j]),
                             '{0:1.5g}'.format(atb_cst['pvalues'].rowmeta['feature_size'][i]),
#                             '{0:1.5g}'.format(atb_cst['pvalues'].columnmeta['cluster_size'][j]),
                             '{0:1.5g}'.format(background_size),
#                             '{0:1.5g}'.format(atb_cst['intersection'].matrix[i,j]),
#                             '{0:1.5g}'.format(atb_cst['union'].matrix[i,j]),
                             atb_cst['pvalues'].rowmeta['preferred_correlated_rowstat'][i],
                             atb_cst['pvalues'].rowmeta['correlated_features'][i]]
                fw.write('\t'.join(writelist) + '\n')
                if atb_cst['correlation_sign'].matrix[i,j] > 0 and atb_cst['is_significant'].matrix[i,j] > 0:
                    fwf.write('\t'.join(writelist) + '\n')
    # plot results
    gp = gene_proj.tolabels(rowlabels=atb_cst['pvalues'].columnlabels.copy())
    for feature, pvalues, signs, significances in zip(atb_cst['pvalues'].rowlabels, atb_cst['pvalues'].matrix, atb_cst['correlation_sign'].matrix, atb_cst['is_significant'].matrix):
        if significances.any():
            plt.figure()
            plt.scatter(gp.matrix[:,0], gp.matrix[:,1], c=np.exp(-1.0*signs*np.log10(pvalues)))
            plt.title(feature)
            plt.show()