# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import copy
import pickle
import datasetIO
from dataclasses import datamatrix as DataMatrix
import mapper
import os
import shutil
import pandas as pd
from matplotlib import pyplot as plt

# load the data
print('loading data...', flush=True)
df = pd.read_csv('../../original_data/pratfelip_symlnk/patient_gene_datamatrix_nanostring_normalized.csv')
gene_atb = DataMatrix(rowname='symbol',
                      rowlabels=df.columns[1:].values,
                      rowmeta={},
                      columnname='geo_accession',
                      columnlabels=df['#'].values,
                      columnmeta={},
                      matrixname='gene_expression_in_tumor_samples_nanostring',
                      matrix=df.values[:,1:].astype('float64').T)
del df

# load the sample metadata
print('loading sample metadata...', flush=True)
df = pd.read_csv('../../original_data/pratfelip_symlnk/patient_sample_characteristics_edited.csv')
assert((df.geo_accession == gene_atb.columnlabels).all())
for field in df.columns:
	if np.in1d(df[field].values, [0, 1]).all():
		gene_atb.columnmeta[field] = df[field].values.astype('bool')
	else:
		gene_atb.columnmeta[field] = df[field].values
gene_atb.columnname = 'patient_id'
gene_atb.columnlabels = gene_atb.columnmeta['patient_id'].copy()
del gene_atb.columnmeta['patient_id']

# shuffle the data
print('shuffling data...', flush=True)
gene_atb.reorder(np.random.permutation(gene_atb.shape[0]), 0)
gene_atb.reorder(np.random.permutation(gene_atb.shape[1]), 1)

# check datamatrix
print('printing datamatrix properties...', flush=True)
print(gene_atb, flush=True)
print('gene_atb.rowlabels.shape', gene_atb.rowlabels.shape, flush=True)
print('gene_atb.columnlabels.shape', gene_atb.columnlabels.shape, flush=True)
print('rowmeta...', flush=True)
for k,v in gene_atb.rowmeta.items():
    print(k, v.dtype, v.shape, v[:3], flush=True)
print('columnmeta...', flush=True)
for k,v in gene_atb.columnmeta.items():
    print(k, v.dtype, v.shape, v[:3], flush=True)

# add hgnc metadata
print('adding hgnc metadata...', flush=True)
hgncmetadata = mapper.annotate_genes(field='symbol', values=gene_atb.rowlabels, metadatapath='../../mappings/hgnc/hgnc_20181016_complete_set.txt', drop_duplicates=True)
gene_atb.rowmeta.update(hgncmetadata)
tobediscarded = np.logical_or(gene_atb.rowmeta['entrez_id'] == 'nan', gene_atb.rowmeta['ensembl_gene_id'] == 'nan')
gene_atb.discard(tobediscarded, 0)
gene_atb.rowname = 'ensembl_gene_id'
gene_atb.rowlabels = gene_atb.rowmeta['ensembl_gene_id'].copy()
del gene_atb.rowmeta['ensembl_gene_id']
print(gene_atb, flush=True)
print('rowmeta...', flush=True)
for k,v in gene_atb.rowmeta.items():
    print(k, v.dtype, v.shape, v[:3], flush=True)

# discard pseudogenes
print('discarding pseudogenes...', flush=True)
print(np.unique(gene_atb.rowmeta['locus_type']).tolist(), flush=True)
tobediscarded = ~np.in1d(gene_atb.rowmeta['locus_type'], ['RNA, long non-coding', 'RNA, micro', 'T cell receptor gene', 'gene with protein product', 'immunoglobulin gene', 'protocadherin'])
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50); plt.savefig('../../original_data/pratfelip_symlnk/hist_5cols_normalizednanostring.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10); plt.savefig('../../original_data/pratfelip_symlnk/hist_5rows_normalizednanostring_zscored.png', transparent=True, pad_inches=0, dpi=300)

# normalize samples
#print('applying median shift normalization using housekeeping genes...', flush=True)
# want to eliminate housekeeping gene fold changes (after log2, housekeeping gene shifts) between samples
# use median sample as reference sample
# compute housekeeping gene shifts relative to reference sample
# subtract housekeeping gene shifts
# note, this corresponds to dividing counts by a sample specific scaling factors
median_sample = np.median(gene_atb.matrix, 1)
#median_shift_from_median = np.median(gene_atb.matrix - median_sample.reshape(-1,1), 0)
#median_shift_from_median = np.median((gene_atb.matrix - median_sample.reshape(-1,1))[gene_atb.rowmeta['type'] == 'Housekeeping',:], 0)
#print(median_shift_from_median, flush=True)
#gene_atb.matrix -=  median_shift_from_median.reshape(1,-1)
#gene_atb.rowmeta['median_sample_ref'] = median_sample.copy()
#gene_atb.matrixname += '_medianshift'

# standardize rows
print('standardizing rows...', flush=True)
row_mean = gene_atb.matrix.mean(1, keepdims=True)
row_stdv = gene_atb.matrix.std(1, keepdims=True)
gene_atb.rowmeta['row_mean_ref'] = row_mean.reshape(-1)
gene_atb.rowmeta['row_stdv_ref'] = row_stdv.reshape(-1)
gene_atb.matrix = (gene_atb.matrix - row_mean)/row_stdv
column_mean = gene_atb.matrix.mean(0, keepdims=True)
column_stdv = gene_atb.matrix.std(0, keepdims=True)
#gene_atb.columnmeta['column_mean_ref'] = column_mean.reshape(-1)
#gene_atb.columnmeta['column_stdv_ref'] = column_stdv.reshape(-1)
#gene_atb.matrix = (gene_atb.matrix - column_mean)/column_stdv
print(gene_atb, flush=True)
gene_atb.matrixname += '_zscored'

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50); plt.savefig('../../original_data/pratfelip_symlnk/hist_5cols_normalizednanostring_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(gene_atb.matrix[:5,:].T, 10); plt.savefig('../../original_data/pratfelip_symlnk/hist_5rows_normalizednanostring_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(gene_atb.matrix.reshape(-1), 1000); plt.savefig('../../original_data/pratfelip_symlnk/hist_all_normalizednanostring_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(column_mean.reshape(-1), 10); plt.savefig('../../original_data/pratfelip_symlnk/hist_colmeans_normalizednanostring_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(column_stdv.reshape(-1), 10); plt.savefig('../../original_data/pratfelip_symlnk/hist_colstdvs_normalizednanostring_zscored.png', transparent=True, pad_inches=0, dpi=300)

# save the data
print('saving prepared data...', flush=True)
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/gene_patient_pratfelip_nanostring_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/gene_patient_pratfelip_nanostring_prepared.txt.gz', gene_atb)
with open('../../original_data/pratfelip_symlnk/prepare_specs_and_transform_vars.pickle', 'wb') as fw:
    pickle.dump({'median_sample':median_sample, 'row_mean':row_mean, 'row_stdv':row_stdv, 'column_mean':column_mean, 'column_stdv':column_stdv}, fw)
savefolder = '../../input_data/pratfelip'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/pratfelip_symlnk/gene_patient_pratfelip_nanostring_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/pratfelip_symlnk/gene_patient_pratfelip_nanostring_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

# save transpose
print('saving transposed data...', flush=True)
atb_gene = gene_atb.totranspose()
atb_gene.matrixname += '_transposed'
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared.pickle', atb_gene)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared.txt.gz', atb_gene)
savefolder = '../../input_data/pratfelip_transposed'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, atb_gene)
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
