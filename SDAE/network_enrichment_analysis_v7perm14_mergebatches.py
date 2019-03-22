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
import numpy as np
import pickle

# specifications
datasetparams = [('clinvar', 10000, 10),
#                 ('dbgap_cleaned', 20000, 5),
#                 ('gad', 20000, 5),
                 ('gadhighlevel_cleaned', 20000, 5),
#                 ('gobp', 20000, 5),
#                 ('gocc', 20000, 5),
#                 ('gomf', 20000, 5),
#                 ('gwascatalog_cleaned', 20000, 5),
#                 ('gwasdbdisease_cleaned', 20000, 5),
#                 ('gwasdbphenotype_cleaned', 20000, 5),
#                 ('hpo', 20000, 5),
#                 ('hugenavigator', 20000, 5),
#                 ('humancyc', 20000, 5),
#                 ('kegg', 20000, 5),
#                 ('locate', 20000, 5),
#                 ('locatepredicted', 20000, 5),
#                 ('mgimpo', 20000, 5),
                 ('omim', 20000, 5),
                 ('panther', 20000, 5)]
#                 ('reactome', 20000, 5),
#                 ('wikipathways', 20000, 5)]
analysis_version = 'v7perm14network'

for datasetabbrev, permsperbatch, batches in datasetparams:
    permstotal = batches*permsperbatch
    batch = 0
    if os.path.exists('aligned_matrices_{0}/atb_cst_{1}_test_statistic_values_batch{2!s}_numperm{3!s}.pickle'.format(analysis_version, datasetabbrev, batch, permsperbatch)):
        os.rename('aligned_matrices_{0}/atb_cst_{1}_test_statistic_values_batch{2!s}_numperm{3!s}.pickle'.format(analysis_version, datasetabbrev, batch, permsperbatch), 'aligned_matrices_{0}/atb_cst_{1}_test_statistic_values.pickle'.format(analysis_version, datasetabbrev))
    with open('aligned_matrices_{0}/atb_cst_{1}_pvalues_batch{2!s}_numperm{3!s}.pickle'.format(analysis_version, datasetabbrev, batch, permsperbatch), mode='rb') as fr:
        atb_cst = pickle.load(fr)
    atb_cst.matrix = atb_cst.matrix*(permsperbatch + 1) - 1
    for batch in range(1,batches):
        with open('aligned_matrices_{0}/atb_cst_{1}_pvalues_batch{2!s}_numperm{3!s}.pickle'.format(analysis_version, datasetabbrev, batch, permsperbatch), mode='rb') as fr:
            temp = pickle.load(fr)
        atb_cst.matrix += temp.matrix*(permsperbatch + 1) - 1
    atb_cst.matrix = (atb_cst.matrix + 1)/(permstotal + 1)
    print('dataset:{0}, min:{1:1.5g}, median:{2:1.5g}, mean:{3:1.5g}, max:{4:1.5g}, unresolved_frac:{5:1.5g}'
          .format(datasetabbrev, atb_cst.matrix.min(), np.median(atb_cst.matrix), atb_cst.matrix.mean(), atb_cst.matrix.max(), (atb_cst.matrix < 1/permstotal).sum()/atb_cst.matrix.size))
    with open('aligned_matrices_{0}/atb_cst_{1}_pvalues.pickle'.format(analysis_version, datasetabbrev), mode='wb') as fw:
        pickle.dump(atb_cst, fw)
