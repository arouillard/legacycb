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
#import gzip
import numpy as np
import copy
from machinelearning import featureselection #, datasetselection
import machinelearning.dataclasses as dc
import pickle
import time
#from operator import itemgetter
#import matplotlib.pyplot as plt

# specifications
datasetabbrev = sys.argv[1]
batch = int(sys.argv[2])
numperm = 20000
analysis_version = 'v7perm14network'
feature_selection_test_function = featureselection.univariate_vectorized_permtest

# load inputs
print('working on {0}...'.format(datasetabbrev))
with open('aligned_matrices_{0}/gene_atb_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
    gene_atb = pickle.load(fr)
with open('aligned_matrices_{0}/gene_cst_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
    gene_cst = pickle.load(fr)

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
starttime = time.time()
print('starting {0!s} permutations...'.format(numperm))
atb_cst['test_statistic_values'].matrix, atb_cst['pvalues'].matrix = feature_selection_test_function(X=gene_cst.matrix, Y=gene_atb.matrix, numperm=numperm)
atb_cst['pvalues'].matrix = atb_cst['pvalues'].matrix.T
if batch == 0:
    atb_cst['test_statistic_values'].matrix = atb_cst['test_statistic_values'].matrix.T
else:
    del atb_cst['test_statistic_values']
print('finished permutations in {0:1.5g} minutes.'.format((time.time() - starttime)/60.0))

# save results
print('writing results...')
for layername, layer in atb_cst.items():
    with open('aligned_matrices_{0}/atb_cst_{1}_{2}_batch{3!s}_numperm{4!s}.pickle'.format(analysis_version, datasetabbrev, layername, batch, numperm), mode='wb') as fw:
        pickle.dump(layer, fw)
print('done.')
