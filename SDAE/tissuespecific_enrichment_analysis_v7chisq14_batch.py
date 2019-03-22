# -*- coding: utf-8 -*-
"""
Clinical Outcome Classifier
get scores for individual features
multiple hypothesis correction by dataset or altogether?
what is the best test?
"""

import sys
custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
#custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
#               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
#               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
#               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

#import os
import gzip
import numpy as np
import copy
from machinelearning import datasetselection, featureselection
import machinelearning.dataclasses as dc
import pickle
#from operator import itemgetter
#import matplotlib.pyplot as plt

# load dataset metadata
print('loading dataset metadata...')
dataset_info = datasetselection.finddatasets(getalllevels=True)
included_datasetabbrevs = {'clinvar', 'dbgap_cleaned', 'gad', 'gadhighlevel_cleaned', 'gobp', 'gocc', 'gomf', 'gwascatalog_cleaned', 'gwasdbdisease_cleaned', 'gwasdbphenotype_cleaned', 'hpo', 'hugenavigator', 'humancyc', 'kegg', 'locate', 'locatepredicted', 'mgimpo', 'omim', 'panther', 'reactome', 'wikipathways'}
excluded_datasetabbrevs = set(dataset_info.keys()).difference(included_datasetabbrevs)
for datasetabbrev in excluded_datasetabbrevs:
    del dataset_info[datasetabbrev]
    
# specifications
datasetabbrev = sys.argv[1]
datasetinfo = dataset_info[datasetabbrev]
analysis_version = 'v7chisq14tissuespecific'
feature_selection_test_function = featureselection.univariate_vectorized_fisherexacttest
feature_selection_test_name = 'fisher'

# load inputs
print('working on {0}...'.format(datasetabbrev))
with open('aligned_matrices_{0}/gene_atb_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
    gene_atb = pickle.load(fr)
with open('aligned_matrices_{0}/gene_cst_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
    gene_cst = pickle.load(fr)
gene_atb.matrix = gene_atb.matrix.astype('float64')
gene_cst.matrix = gene_cst.matrix.astype('float64')
gene_atb.updatedtypeattribute()
gene_cst.updatedtypeattribute()

# initialize outputs
atb_cst = {}
atb_cst['test_statistic_values'] = dc.datamatrix(rowname=gene_atb.columnname,
                                                 rowlabels=gene_atb.columnlabels.copy(),
                                                 rowmeta=copy.deepcopy(gene_atb.columnmeta),
                                                 columnname=gene_cst.columnname,
                                                 columnlabels=gene_cst.columnlabels.copy(),
                                                 columnmeta=copy.deepcopy(gene_cst.columnmeta),
                                                 matrixname='atb_cluster_correlation',
                                                 matrix=np.zeros((gene_atb.shape[1], gene_cst.shape[1]), dtype='float64'))
atb_cst['pvalues'] = copy.deepcopy(atb_cst['test_statistic_values'])

# computation
print('starting computation...')
atb_cst['test_statistic_values'].matrix, atb_cst['pvalues'].matrix = feature_selection_test_function(X=gene_atb.matrix, Y=gene_cst.matrix)

# compute other values
atb_cst['correlation_sign'] = copy.deepcopy(atb_cst['test_statistic_values'])
atb_cst['pvalues_corrected'] = copy.deepcopy(atb_cst['test_statistic_values'])
atb_cst['is_significant'] = copy.deepcopy(atb_cst['test_statistic_values'])
atb_cst['intersection'] = copy.deepcopy(atb_cst['test_statistic_values'])
atb_cst['union'] = copy.deepcopy(atb_cst['test_statistic_values'])
atb_cst['jaccard'] = copy.deepcopy(atb_cst['test_statistic_values'])
atb_cst['cosine'] = copy.deepcopy(atb_cst['test_statistic_values'])

atb_cst['correlation_sign'].matrix = np.sign(atb_cst['test_statistic_values'].matrix)
atb_cst['is_significant'].matrix, atb_cst['pvalues_corrected'].matrix = featureselection.multiple_hypothesis_testing_correction(atb_cst['pvalues'].matrix, alpha=0.05, method='fdr_by')
atb_cst['intersection'].matrix = (gene_atb.matrix.T).dot(gene_cst.matrix)
atb_cst['union'].matrix = gene_atb.columnmeta['feature_size'].reshape(-1,1) + gene_cst.columnmeta['cluster_size'].reshape(1,-1) - atb_cst['intersection'].matrix
atb_cst['jaccard'].matrix = atb_cst['intersection'].matrix/atb_cst['union'].matrix
atb_cst['cosine'].matrix = atb_cst['intersection'].matrix/np.sqrt(gene_atb.columnmeta['feature_size'].reshape(-1,1)*gene_cst.columnmeta['cluster_size'].reshape(1,-1))

# update dtypes
for layername in atb_cst:
    if layername != 'is_significant':
        atb_cst[layername].matrix = atb_cst[layername].matrix.astype('float32')
    atb_cst[layername].updatedtypeattribute()

# save results
print('saving results...')
for layername, layer in atb_cst.items():
    with open('univariate_feature_importance_{0}/{1}_{2}.pickle'.format(analysis_version, datasetabbrev, layername), mode='wb') as fw:
        pickle.dump(layer, fw)
with gzip.open('univariate_feature_importance_{0}/{1}.txt.gz'.format(analysis_version, datasetabbrev), mode='wt', encoding="utf-8", errors="surrogateescape") as fw, \
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
                 'cluster_size',
                 'background_size',
                 'intersection',
                 'union',
                 'jaccard',
                 'cosine',
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
                         '{0:1.5g}'.format(atb_cst['pvalues'].columnmeta['cluster_size'][j]),
                         '{0:1.5g}'.format(gene_atb.shape[0]),
                         '{0:1.5g}'.format(atb_cst['intersection'].matrix[i,j]),
                         '{0:1.5g}'.format(atb_cst['union'].matrix[i,j]),
                         '{0:1.5g}'.format(atb_cst['jaccard'].matrix[i,j]),
                         '{0:1.5g}'.format(atb_cst['cosine'].matrix[i,j]),
                         atb_cst['pvalues'].rowmeta['preferred_correlated_rowstat'][i],
                         atb_cst['pvalues'].rowmeta['correlated_features'][i]]
            fw.write('\t'.join(writelist) + '\n')
            if atb_cst['correlation_sign'].matrix[i,j] > 0 and atb_cst['is_significant'].matrix[i,j] > 0:
                fwf.write('\t'.join(writelist) + '\n')
print('done.')
