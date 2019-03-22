# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import pandas as pd
import copy
import datasetIO
import os
import shutil
from dataclasses import datamatrix as DataMatrix


# load the data
print('loading dataset...', flush=True)
dataset = datasetIO.load_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical.pickle')
dataset.reorder(np.argsort(dataset.rowmeta['geo_accession']), 0)
dataset.columnname += '_or_deconv_variable'
dataset.matrixname += '_and_deconv_variables'
dataset.columnmeta['is_deconv_variable'] = np.zeros(dataset.shape[1], dtype='bool')
print(dataset, flush=True)

# load and append deconv variables
print('loading and appending deconv variables...', flush=True)
deconv_methods = ['epic', 'MCPCounter', 'quantiseq', 'timer']
for deconv_method in deconv_methods:
    print('working on deconv_method: {0}...'.format(deconv_method), flush=True)
    df = pd.read_csv('../../original_data/pratfelip_symlnk/deconvoluted/deconvoluted_{0}.csv'.format(deconv_method), index_col=False)
    df.sort_values(by='Unnamed: 0', inplace=True)
    if df.shape[0] == dataset.shape[0] and (df['Unnamed: 0'].values == dataset.rowmeta['geo_accession']).all():
        # create datamatrix of deconv variables
        print('creating datamatrix of deconv variables...', flush=True)
        deconv_dataset = DataMatrix(rowname=dataset.rowname,
                                    rowlabels=dataset.rowlabels.copy(),
                                    rowmeta=copy.deepcopy(dataset.rowmeta),
                                    columnname=dataset.columnname,
                                    columnlabels=df.columns[1:].values + '|{0}'.format(deconv_method),
                                    columnmeta={'cell_type':df.columns[1:].values, 'deconv_method':np.full(df.shape[1]-1, deconv_method, dtype='object'), 'variable_type':np.full(df.shape[1]-1, 'deconv_variable', dtype='object'), 'is_gene':np.zeros(df.shape[1]-1, dtype='bool'), 'is_clinical_variable':np.zeros(df.shape[1]-1, dtype='bool'), 'is_deconv_variable':np.ones(df.shape[1]-1, dtype='bool')},
                                    matrixname='deconv_variables_for_tumor_samples',
                                    matrix=(df.values[:,1:]).astype('float64'))
        print(deconv_dataset, flush=True)
        # zscore deconv variables
        print('standardizing deconv variables...', flush=True)
        deconv_mean = deconv_dataset.matrix.mean(0, keepdims=True)
        deconv_stdv = deconv_dataset.matrix.std(0, keepdims=True)
        deconv_stdv[deconv_stdv==0] = 1
        deconv_dataset.matrix = (deconv_dataset.matrix - deconv_mean)/deconv_stdv
        # append deconv variables
        print('appending deconv variables...', flush=True)
        dataset.append(deconv_dataset, 1)
        print(dataset, flush=True)
    else:
        print('WARNING! Rows do not match!', flush=True)

# print row metadata
print('printing row metadata...', flush=True)
for k,v in dataset.rowmeta.items():
    print(k, v.shape, v.dtype, v[:3], flush=True)

# print column metadata
print('printing column metadata...', flush=True)
for k,v in dataset.columnmeta.items():
    print(k, v.shape, v.dtype, v[:3], flush=True)


# save the data
print('saving data with deconv variables...', flush=True)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_plus_deconv.pickle', dataset)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_plus_deconv.txt.gz', dataset)
savefolder = '../../input_data/pratfelip_transposed_plus_clinical_plus_deconv'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, dataset)
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_plus_deconv.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_plus_deconv.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

# discard genes
print('discarding genes...', flush=True)
dataset.discard(dataset.columnmeta['is_gene'], 1)
print(dataset, flush=True)

# save the data
print('saving data with only clinical and deconv variables...', flush=True)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv.pickle', dataset)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv.txt.gz', dataset)
savefolder = '../../input_data/pratfelip_clinical_and_deconv'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, dataset)
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
