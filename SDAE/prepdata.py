# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import numpy as np
import pickle
import machinelearning.datasetselection as ds
import os

# load the data
gene_atb = ds.loaddatamatrix('data/original_data/gene_attribute_matrix_cleaned.txt.gz',
                             rowname='gene',
                             columnname='atb',
                             matrixname='gene_atb_associations',
                             skiprows=3,
                             skipcolumns=3,
                             delimiter='\t',
                             dtype='float32',
                             getmetadata=True, # need to fix False case
                             getmatrix=True)

# shuffle the data
gene_atb.reorder(np.random.permutation(gene_atb.shape[0]), 0)
gene_atb.reorder(np.random.permutation(gene_atb.shape[1]), 1)

# standardize the data
row_mean = gene_atb.matrix.mean(1)
row_stdv = gene_atb.matrix.std(1)
standardized_row_mean = (row_mean - row_mean.mean())/row_mean.std()
standardized_row_stdv = (row_stdv - row_stdv.mean())/row_stdv.std()
gene_atb.matrix = (gene_atb.matrix - row_mean[:,np.newaxis])/row_stdv[:,np.newaxis]
column_mean = gene_atb.matrix.mean(0)
column_stdv = gene_atb.matrix.std(0)
adjusted_row_mean = standardized_row_mean*column_stdv.mean() + column_mean.mean()
adjusted_row_stdv = standardized_row_stdv*column_stdv.mean() + column_mean.mean()
gene_atb.matrix = np.concatenate((gene_atb.matrix, adjusted_row_mean[:,np.newaxis], adjusted_row_stdv[:,np.newaxis]), 1)
gene_atb.columnlabels = np.append(gene_atb.columnlabels, np.array(['adjusted_row_mean', 'adjusted_row_stdv'], dtype='object'))
gene_atb.updateshapeattribute()
gene_atb.updatesizeattribute()
#column_mean = gene_atb.matrix.mean(0)
#column_stdv = gene_atb.matrix.std(0)
#gene_atb.matrix = (gene_atb.matrix - column_mean[np.newaxis,:])/column_stdv[np.newaxis,:]

# split the data
test_fraction = 0.1
tobepopped = np.random.permutation(gene_atb.shape[0]) < round(test_fraction*gene_atb.shape[0])
gene_atb_test = gene_atb.pop(tobepopped, 0)
valid_fraction = 0.1
tobepopped = np.random.permutation(gene_atb.shape[0]) < round(valid_fraction*gene_atb.shape[0])
gene_atb_valid = gene_atb.pop(tobepopped, 0)
gene_atb_train = gene_atb
test_examples = gene_atb_test.shape[0]
valid_examples = gene_atb_valid.shape[0]
train_examples = gene_atb_train.shape[0]
del gene_atb, tobepopped

# save the data
if not os.path.exists('data/prepared_data'):
    os.mkdir('data/prepared_data')
with open('data/prepared_data/test.pickle', 'wb') as fw:
    pickle.dump(gene_atb_test, fw)
with open('data/prepared_data/valid.pickle', 'wb') as fw:
    pickle.dump(gene_atb_valid, fw)
with open('data/prepared_data/train.pickle', 'wb') as fw:
    pickle.dump(gene_atb_train, fw)
with open('data/prepared_data/specs_and_transform_vars.pickle', 'wb') as fw:
    pickle.dump((column_mean, column_stdv, test_fraction, valid_fraction, test_examples, valid_examples, train_examples), fw)
