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

import os
#import gzip
import numpy as np
import copy
from machinelearning import datasetselection, featureselection
#import machinelearning.dataclasses as dc
import pickle
from operator import itemgetter
#import matplotlib.pyplot as plt

# load clusters
with open('/GWD/bioinfo/projects/cb01/harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_nonspecific.pickle', 'rb') as fr:
    gene_cluster = pickle.load(fr)

# select datasets
dataset_info = datasetselection.finddatasets(getalllevels=True)
included_datasetabbrevs = {'clinvar', 'dbgap_cleaned', 'gad', 'gadhighlevel_cleaned', 'gobp', 'gocc', 'gomf', 'gwascatalog_cleaned', 'gwasdbdisease_cleaned', 'gwasdbphenotype_cleaned', 'hpo', 'hugenavigator', 'humancyc', 'kegg', 'locate', 'locatepredicted', 'mgimpo', 'omim', 'panther', 'reactome', 'wikipathways'}
excluded_datasetabbrevs = set(dataset_info.keys()).difference(included_datasetabbrevs)
for datasetabbrev in excluded_datasetabbrevs:
    del dataset_info[datasetabbrev]
    
# analysis version
analysis_version = 'v7chisq14tissue'
if not os.path.exists('datasets_in_progress_{0}'.format(analysis_version)):
    os.mkdir('datasets_in_progress_{0}'.format(analysis_version))
if not os.path.exists('feature_groups_{0}'.format(analysis_version)):
    os.mkdir('feature_groups_{0}'.format(analysis_version))
if not os.path.exists('univariate_feature_importance_{0}'.format(analysis_version)):
    os.mkdir('univariate_feature_importance_{0}'.format(analysis_version))
if not os.path.exists('aligned_matrices_{0}'.format(analysis_version)):
    os.mkdir('aligned_matrices_{0}'.format(analysis_version))

# parameters
feature_selection_test_function = featureselection.univariate_chisquare
feature_selection_test_name = 'chisquare'
similarity_metric = 'cosine'
similarity_threshold = np.sqrt(0.5)

# iterate over datasets
for datasetabbrev, datasetinfo in dataset_info.items():
    # just work with pathways for testing/debugging the pipeline
#    if datasetabbrev not in {'clinvar'}: #{'kegg', 'panther', 'reactome', 'wikipathways'}:
#        continue
    if os.path.exists('datasets_in_progress_{0}/{1}.txt'.format(analysis_version, datasetabbrev)):
        continue
    with open('datasets_in_progress_{0}/{1}.txt'.format(analysis_version, datasetabbrev), mode='wt', encoding="utf-8", errors="surrogateescape") as fw:
        fw.write('working on {}...'.format(datasetabbrev))
    print('working on {}...'.format(datasetabbrev))
    # load dataset
    gene_atb = datasetselection.loaddatamatrix(datasetpath=datasetinfo['path'],
                                         rowname='gene',
                                         columnname='atb',
                                         matrixname='gene_atb_associations',
                                         skiprows=3,
                                         skipcolumns=3,
                                         delimiter='\t',
                                         dtype='float64',
                                         getmetadata=True, # need to fix False case
                                         getmatrix=True)
    # check binary
    if set(np.unique(gene_atb.matrix)) != {0, 1}:
        print('warning: converting matrix to binary values')
    gene_atb.matrix = gene_atb.matrix != 0
    gene_atb.updatedtypeattribute()
    # compute feature similarity
    atb_atb = gene_atb.tosimilarity(axis=1, metric=similarity_metric)
    # align with clusters
    commongenes = gene_atb.rowlabels[np.in1d(gene_atb.rowlabels, gene_cluster.rowlabels)]
    gene_atb = gene_atb.tolabels(rowlabels=commongenes)
    gene_cst = gene_cluster.tolabels(rowlabels=commongenes)
    del commongenes
    # discard bad columns
    tobediscarded = np.logical_or.reduce(((gene_atb.matrix != 0).sum(axis=0) < 3, (gene_atb.matrix == 0).sum(axis=0) < 3, np.isnan(gene_atb.matrix).any(axis=0)))
    gene_atb.discard(tobediscarded, axis=1)
    tobediscarded = np.logical_or.reduce(((gene_cst.matrix != 0).sum(axis=0) < 3, (gene_cst.matrix == 0).sum(axis=0) < 3, np.isnan(gene_cst.matrix).any(axis=0)))
    gene_cst.discard(tobediscarded, axis=1)
    if gene_atb.shape[0] == 0 or gene_atb.shape[1] == 0 or gene_cst.shape[0] == 0 or gene_cst.shape[1] == 0:
        continue
    # arbitrary prioritization to break redundancyindex ties later
    gene_atb.columnmeta['arbitrary_pvalues'] = featureselection.univariate_chisquare(X=gene_atb.matrix, Y=gene_cst.matrix[:,0])[1]
    tobediscarded = np.isnan(gene_atb.columnmeta['arbitrary_pvalues'])
    gene_atb.discard(tobediscarded, axis=1)
    if gene_atb.shape[0] == 0 or gene_atb.shape[1] == 0:
        continue
    # discard redundant features
    rowstatpreferredorder = np.array(['mean', 'stdv'], dtype='object')
    atb_atb = atb_atb.tolabels(gene_atb.columnlabels, gene_atb.columnlabels)
    atb_atb.rowmeta = copy.deepcopy(gene_atb.columnmeta)
    atb_atb.columnmeta = copy.deepcopy(gene_atb.columnmeta)
    redundancyindex = (np.abs(atb_atb.matrix) > similarity_threshold).sum(1).astype('float64')
    for i, rowstat in enumerate(rowstatpreferredorder):
        if rowstat in atb_atb.rowlabels:
            redundancyindex[atb_atb.rowlabels==rowstat] += 1/(2 + i)
    table = list(zip(np.arange(atb_atb.shape[0], dtype='int64'), redundancyindex.copy(), atb_atb.rowmeta['arbitrary_pvalues'].copy(), atb_atb.rowlabels.copy()))
    table.sort(key=itemgetter(3))
    table.sort(key=itemgetter(2))
    table.sort(key=itemgetter(1), reverse=True)
#    for t in table:
#        print('{0!s}, {1!s}, {2:1.5g}, {3}'.format(t[0], t[1], t[2], t[3]))
    sortedindices = np.array([t[0] for t in table], dtype='int64')
    atb_atb.reorder(sortedindices, axis=0)
    atb_atb.reorder(sortedindices, axis=1)
    gene_atb = gene_atb.tolabels(columnlabels=atb_atb.columnlabels.copy())
    tobediscarded = np.zeros(gene_atb.shape[1], dtype='bool')
    gene_atb.columnmeta['correlated_features'] = np.empty(gene_atb.shape[1], dtype='object')
    gene_atb.columnmeta['preferred_correlated_rowstat'] = np.full(gene_atb.shape[1], '', dtype='object')
    with open('feature_groups_{0}/{1}.txt'.format(analysis_version, datasetabbrev), mode='wt', encoding="utf-8", errors="surrogateescape") as fw:
        for i in range(atb_atb.shape[0]):
            if ~tobediscarded[i]:
                tbd = np.abs(atb_atb.matrix[i,:]) > similarity_threshold
                tbd = np.logical_and(tbd, ~tobediscarded) # just what's new
                tbd[:i+1] = False
                gene_atb.columnmeta['correlated_features'][i] = '|'.join(atb_atb.columnlabels[tbd].tolist())
                if np.in1d(rowstatpreferredorder, np.insert(atb_atb.columnlabels[tbd], 0, atb_atb.columnlabels[i])).any():
                    gene_atb.columnmeta['preferred_correlated_rowstat'][i] = rowstatpreferredorder[np.in1d(rowstatpreferredorder, np.insert(atb_atb.columnlabels[tbd], 0, atb_atb.columnlabels[i])).nonzero()[0][0]]
                    gene_atb.matrix[:,i] = gene_atb.select([],gene_atb.columnmeta['preferred_correlated_rowstat'][i])
#                elif tbd.any():
#                    mrg = tbd.copy()
#                    mrg[i] = True
#                    wgt = atb_atb.matrix[i,mrg]
#                    wgt = wgt/np.sum(np.abs(wgt))
#                    gene_atb.matrix[:,i] = (gene_atb.matrix[:,mrg]*wgt[np.newaxis,:]).sum(1)
                writelist = [atb_atb.rowlabels[i]]
                for j in tbd.nonzero()[0]:
                    writelist.append('{0}|{1:1.5g}'.format(atb_atb.columnlabels[j], atb_atb.matrix[i,j]))
                fw.write('\t'.join(writelist) + '\n')
                tobediscarded = np.logical_or(tobediscarded, tbd)
    gene_atb.discard(tobediscarded, axis=1)
    if gene_atb.shape[0] == 0 or gene_atb.shape[1] == 0:
        continue
    gene_atb.columnmeta['feature_size'] = (gene_atb.matrix != 0).sum(0)
    gene_cst.columnmeta['cluster_size'] = gene_cst.matrix.sum(0)
    del tobediscarded, rowstatpreferredorder, atb_atb, redundancyindex, i, rowstat, table, sortedindices, tbd, writelist
    # save matrices
    with open('aligned_matrices_{0}/gene_atb_{1}.pickle'.format(analysis_version, datasetabbrev), mode='wb') as fw:
        pickle.dump(gene_atb, fw)
    with open('aligned_matrices_{0}/gene_cst_{1}.pickle'.format(analysis_version, datasetabbrev), mode='wb') as fw:
        pickle.dump(gene_cst, fw)
    del gene_atb, gene_cst
    '''
    # calculate significance of feature-cluster correlations
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
    atb_cst['pvalues_corrected'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['class_mean_difference'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['correlation_sign'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['is_significant'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['intersection'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['union'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['jaccard'] = copy.deepcopy(atb_cst['test_statistic_values'])
    atb_cst['phi'] = copy.deepcopy(atb_cst['test_statistic_values'])
    anyrowstats = gene_atb.columnmeta['isrowstat'].any()
    anynonstats = (~gene_atb.columnmeta['isrowstat']).any()
    for j, cluster_label in enumerate(gene_cst.columnlabels):
        if np.mod(j, 1000) == 0:
            print('testing cluster {0!s} of {1!s}: {2}...'.format(j, gene_cst.shape[1], cluster_label))
        if anynonstats:
            atb_cst['test_statistic_values'].matrix[~gene_atb.columnmeta['isrowstat'],j], atb_cst['pvalues'].matrix[~gene_atb.columnmeta['isrowstat'],j], atb_cst['phi'].matrix[~gene_atb.columnmeta['isrowstat'],j] = feature_selection_test_function(X=gene_atb.matrix[:,~gene_atb.columnmeta['isrowstat']], Y=gene_cst.matrix[:,j])
        if anyrowstats:
            atb_cst['test_statistic_values'].matrix[gene_atb.columnmeta['isrowstat'],j], atb_cst['pvalues'].matrix[gene_atb.columnmeta['isrowstat'],j] = featureselection.univariate_utest(X=gene_atb.matrix[:,gene_atb.columnmeta['isrowstat']], Y=gene_cst.matrix[:,j])
        atb_cst['class_mean_difference'].matrix[:,j] = featureselection.class_difference(X=gene_atb.matrix, Y=gene_cst.matrix[:,j])[0]
    atb_cst['correlation_sign'].matrix = np.sign(atb_cst['class_mean_difference'].matrix)
    atb_cst['is_significant'].matrix, atb_cst['pvalues_corrected'].matrix = featureselection.multiple_hypothesis_testing_correction(atb_cst['pvalues'].matrix, alpha=0.05, method='fdr_by')
    atb_cst['intersection'].matrix = np.float64(gene_atb.matrix.T != 0).dot(gene_cst.matrix)
    atb_cst['union'].matrix = gene_atb.columnmeta['feature_size'].reshape(-1,1) + gene_cst.columnmeta['cluster_size'].reshape(1,-1) - atb_cst['intersection'].matrix
    atb_cst['jaccard'].matrix = atb_cst['intersection'].matrix/atb_cst['union'].matrix
    # save results
    for layername, layer in atb_cst.items():
        with open('univariate_feature_importance_{0}/{1}_{2}.pickle'.format(analysis_version, datasetabbrev, layername), mode='wb') as fw:
            pickle.dump(layer, fw)
    with open('univariate_feature_importance_{0}/{1}.txt'.format(analysis_version, datasetabbrev), mode='wt', encoding="utf-8", errors="surrogateescape") as fw:
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
                     'phi',
                     'preferred_correlated_rowstat',
                     'correlated_features']
        fw.write('\t'.join(writelist) + '\n')
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
                             '{0:1.5g}'.format(atb_cst['phi'].matrix[i,j]),
                             atb_cst['pvalues'].rowmeta['preferred_correlated_rowstat'][i],
                             atb_cst['pvalues'].rowmeta['correlated_features'][i]]
                fw.write('\t'.join(writelist) + '\n')
    '''
