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
import pickle
from machinelearning import datasetselection

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
#results_layers = {'test_statistic_values', 'pvalues', 'correlation_sign', 'is_significant'}
results_layers = {'test_statistic_values', 'is_significant'}

# iterate over datasets
with open('univariate_feature_importance_{0}/significant_feature_counts.txt'.format(analysis_version), mode='wt', buffering=1) as fw, open('univariate_feature_importance_{0}/significant_features.txt'.format(analysis_version), mode='wt') as fw2:
    fw.write('\t'.join(['dataset_name', 'dataset_abbreviation', 'significant_features']) + '\n')
    fw2.write('\t'.join(['dataset_name', 'dataset_abbreviation', 'significant_feature']) + '\n')
    for datasetabbrev, datasetinfo in dataset_info.items():
        # just work with pathways for testing/debugging the pipeline
    #    if 'gtex_overexpression' not in datasetabbrev: # datasetabbrev not in {'kegg', 'panther', 'reactome', 'wikipathways'}:
    #        continue
        if not os.path.exists('univariate_feature_importance_{0}/{1}_pvalues.pickle'.format(analysis_version, datasetabbrev)):
            continue
        print('working on {0}...'.format(datasetabbrev), flush=True)
        # load pvalues, test statistic values, etc.
        print('loading results...', flush=True)
        atb_cst = {}
        for layer in results_layers:
            with open('univariate_feature_importance_{0}/{1}_{2}.pickle'.format(analysis_version, datasetabbrev, layer), mode='rb') as fr:
                atb_cst[layer] = pickle.load(fr)
    #    minp = atb_cst['pvalues'].matrix[atb_cst['pvalues'].matrix > 0].min()
    #    atb_cst['pvalues'].matrix[atb_cst['pvalues'].matrix == 0] = minp
        # keep only significant features (include effect size cutoff)
        atb_cst['is_significant'].matrix = np.logical_and(atb_cst['is_significant'].matrix, atb_cst['test_statistic_values'].matrix > 0.01)
        num_sig_atbs = atb_cst['is_significant'].matrix.any(1).sum()
        sig_atbs = atb_cst['is_significant'].rowlabels[atb_cst['is_significant'].matrix.any(1)]
        print('significant features: {0!s}'.format(num_sig_atbs), flush=True)
        fw.write('\t'.join([datasetinfo['name'], datasetinfo['abbreviation'], str(num_sig_atbs)]) + '\n')
        for sig_atb in sig_atbs:
            fw2.write('\t'.join([datasetinfo['name'], datasetinfo['abbreviation'], sig_atb]) + '\n')
print('done.')
