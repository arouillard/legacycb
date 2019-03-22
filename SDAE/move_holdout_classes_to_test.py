# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import os
import numpy as np
import datasetIO

'''
folder_name = 'GTEXv6'
source_path = 'data/prepared_data/{0}/fat'.format(folder_name)
target_path = 'data/prepared_data/{0}_hoc/fat'.format(folder_name)
os.makedirs(target_path)
os.makedirs(target_path.replace('data/prepared_data', 'results/autoencoder'))

holdout_field = 'general_tissue'
holdout_values = ['Adrenal Gland', 'Thyroid', 'Kidney']

# load data
test = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'test'))
print('test')
print(test)
partitions = ['train', 'valid']
for partition in partitions:
    dm = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, partition))
    print(partition)
    print(dm)
    test.append(dm.pop(np.in1d(dm.rowmeta[holdout_field], holdout_values), 0), 0)
    print(partition)
    print(dm)
    print('test')
    print(test)
    datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, partition), dm)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'test'), test)
'''

'''
import copy

folder_name = 'GTEXv6'
source_path = 'data/prepared_data/{0}/fat'.format(folder_name)
target_path = 'data/prepared_data/{0}_hoc/fat'.format(folder_name)
os.makedirs(target_path)
#os.makedirs(target_path.replace('data/prepared_data', 'results/autoencoder'))

holdout_field = 'general_tissue'
partitions = ['train', 'valid', 'test']
dataset = {}
for partition in partitions:
    dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, partition))
    if 'all' not in dataset:
        dataset['all'] = copy.deepcopy(dataset[partition])
    else:
        dataset['all'].append(dataset[partition], 0)

tissue_gene = copy.deepcopy(dataset['all'])
tissue_gene.rowlabels = tissue_gene.rowmeta[holdout_field].copy()
tissue_gene.rowname = holdout_field
tissue_gene.merge(0)
tissue_gene.discard(tissue_gene.rowlabels == '-666', 0)
tissue_gene.cluster(0, 'correlation')
tissue_tissue = tissue_gene.tosimilarity(0, 'pearson')
S = tissue_tissue.matrix.copy()
S[np.eye(S.shape[0], dtype='bool')] = 0
for t, s in zip(tissue_tissue.rowlabels, tissue_tissue.matrix):
    si = np.argsort(s**2)[::-1]
    print('  '.join([t[:4]] + ['{0},{1:1.2g}'.format(x[:4],y) for x,y in zip(tissue_tissue.rowlabels[si[1:6]], s[si[1:6]])]))
selected_tissues = ['Adrenal Gland', 'Thyroid', 'Liver', 'Pituitary']
sel_sel = tissue_tissue.tolabels(selected_tissues, selected_tissues)
'''

'''
folder_name = 'GTEXv6'
source_path = 'data/prepared_data/{0}/fat'.format(folder_name)
target_path = 'data/prepared_data/{0}_hoc2/fat'.format(folder_name)
os.makedirs(target_path)
os.makedirs(target_path.replace('data/prepared_data', 'results/autoencoder'))

holdout_field = 'general_tissue'
holdout_values = ['Adrenal Gland', 'Thyroid', 'Liver', 'Pituitary']

# load data
train = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'train'))
valid = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'valid'))
test = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'test'))
print('train')
print(train)
print('valid')
print(valid)
print('test')
print(test)
test.append(train.pop(np.in1d(train.rowmeta[holdout_field], holdout_values), 0), 0)
test.append(valid.pop(np.in1d(valid.rowmeta[holdout_field], holdout_values), 0), 0)
valid.append(test.pop(~np.in1d(test.rowmeta[holdout_field], holdout_values), 0), 0)
print('train')
print(train)
print('valid')
print(valid)
print('test')
print(test)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'train'), train)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'valid'), valid)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'test'), test)
'''

'''
folder_name = 'GTEXv6'
source_path = 'data/prepared_data/{0}/fat'.format(folder_name)
target_path = 'data/prepared_data/{0}_hoc3/fat'.format(folder_name)
os.makedirs(target_path)
os.makedirs(target_path.replace('data/prepared_data', 'results/autoencoder'))

holdout_field = 'general_tissue'
holdout_values = ['Adrenal Gland', 'Thyroid', 'Liver', 'Pituitary']
valid_values = ['Adrenal Gland', 'Liver']
test_values = ['Thyroid', 'Pituitary']

# load data
train = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'train'))
valid = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'valid'))
test = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'test'))
print('train')
print(train)
print('valid')
print(valid)
print('test')
print(test)
test.append(train.pop(np.in1d(train.rowmeta[holdout_field], test_values), 0), 0)
test.append(valid.pop(np.in1d(valid.rowmeta[holdout_field], test_values), 0), 0)
valid.append(train.pop(np.in1d(train.rowmeta[holdout_field], valid_values), 0), 0)
valid.append(test.pop(np.in1d(test.rowmeta[holdout_field], valid_values), 0), 0)
train.append(valid.pop(~np.in1d(valid.rowmeta[holdout_field], holdout_values), 0), 0)
train.append(test.pop(~np.in1d(test.rowmeta[holdout_field], holdout_values), 0), 0)
print('train')
print(train)
print('valid')
print(valid)
print('test')
print(test)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'train'), train)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'valid'), valid)
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'test'), test)
'''


folder_name = 'GTEXv6'
source_path = 'data/prepared_data/{0}/fat'.format(folder_name)
target_path = 'data/prepared_data/{0}_tsub/fat'.format(folder_name)
os.makedirs(target_path)
os.makedirs(target_path.replace('data/prepared_data', 'results/autoencoder'))

train = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'train'))
tobediscarded = train.rowmeta['general_tissue'] == '-666'
train.discard(tobediscarded, 0)
Y = train.matrix.copy()
l = train.rowmeta['general_tissue'].copy()
L = np.unique(l)
X = np.float64(l.reshape(-1,1) == L.reshape(1,-1))
X = np.append(X, np.ones((X.shape[0], 1), dtype='float64'), 1)
B, _, rank, singular_values = np.linalg.lstsq(X, Y, rcond=None)
Ypred = X.dot(B)
train.matrix = Y - Ypred
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'train'), train)

valid = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'valid'))
tobediscarded = valid.rowmeta['general_tissue'] == '-666'
valid.discard(tobediscarded, 0)
Y = valid.matrix.copy()
l = valid.rowmeta['general_tissue'].copy()
X = np.float64(l.reshape(-1,1) == L.reshape(1,-1))
X = np.append(X, np.ones((X.shape[0], 1), dtype='float64'), 1)
Ypred = X.dot(B)
valid.matrix = Y - Ypred
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'valid'), valid)

test = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(source_path, 'test'))
tobediscarded = test.rowmeta['general_tissue'] == '-666'
test.discard(tobediscarded, 0)
Y = test.matrix.copy()
l = test.rowmeta['general_tissue'].copy()
X = np.float64(l.reshape(-1,1) == L.reshape(1,-1))
X = np.append(X, np.ones((X.shape[0], 1), dtype='float64'), 1)
Ypred = X.dot(B)
test.matrix = Y - Ypred
datasetIO.save_datamatrix('{0}/{1}.pickle'.format(target_path, 'test'), test)
