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
import mapper
import os
import shutil
import pandas as pd
from matplotlib import pyplot as plt

# load the data
print('loading data...', flush=True)
gene_atb = datasetIO.load_datamatrix('../../original_data/hugolo_symlnk/gene_patient_fpkm_matrix.csv', delimiter=',')
gene_atb.rowname = 'symbol'
gene_atb.columnname = 'patient_id'
gene_atb.matrixname = 'gene_expression_in_tumor_samples_RNAseq_fpkm'

# load the sample metadata
print('loading sample metadata...', flush=True)
df = pd.read_csv('../../original_data/hugolo_symlnk/s1a_patient_sample_characteristics_filtered_edited.csv')
assert((df.patient_id == gene_atb.columnlabels).all())
for field in df.columns:
	if np.in1d(df[field].values, [0, 1]).all():
		gene_atb.columnmeta[field] = df[field].values.astype('bool')
	else:
		gene_atb.columnmeta[field] = df[field].values
del gene_atb.columnmeta['patient_id']

# shuffle the data
print('shuffling data...', flush=True)
gene_atb.reorder(np.random.permutation(gene_atb.shape[0]), 0)
gene_atb.reorder(np.random.permutation(gene_atb.shape[1]), 1)
print(gene_atb, flush=True)

# add hgnc metadata
print('adding hgnc metadata...', flush=True)
hgncmetadata = mapper.annotate_genes(field='symbol', values=gene_atb.rowlabels, metadatapath='../../mappings/hgnc/hgnc_20181016_complete_set.txt', drop_duplicates=True)
gene_atb.rowmeta.update(hgncmetadata)
tobediscarded = np.logical_or(gene_atb.rowmeta['entrez_id'] == 'nan', gene_atb.rowmeta['ensembl_gene_id'] == 'nan')
gene_atb.discard(tobediscarded, 0)
gene_atb.rowname = 'ensembl_gene_id'
gene_atb.rowlabels = gene_atb.rowmeta['ensembl_gene_id'].copy()
del gene_atb.rowmeta['ensembl_gene_id']
print(gene_atb.rowmeta.keys(), flush=True)
print(gene_atb, flush=True)

# discard pseudogenes
print('discarding pseudogenes...', flush=True)
print(np.unique(gene_atb.rowmeta['locus_type']).tolist(), flush=True)
tobediscarded = ~np.in1d(gene_atb.rowmeta['locus_type'], ['RNA, long non-coding', 'RNA, micro', 'T cell receptor gene', 'gene with protein product', 'immunoglobulin gene', 'protocadherin'])
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)

# handle zeros
print('handling zeros...', flush=True)
# dynamic range of fpkm for each sample is a function of library size (mapped reads) and range of gene lengths
# assuming a library size of 1e7 and a ratio of gene lengths of 1e4, upper bound of dynamic range should be 1e11, but this would require lowest expressed gene to also be the longest and highest expressed gene to also be the shortest, unlikely, so expect dynamic range actually 1e10 or 1e9
# then given max fpkm of 1e7, lower bound on detection limit for fpkm should be 1e-4, but unlikely, so actually expect detection limit 1e-3 or 1e-2
# if detection limit = 1e-30 then discard 4623 genes, if 1e-9 then discard 4633, if 1e-4 then discard 4644, if 1e-3 then discard 4650, if 0.01 then discard 4992, if 0.1 then discard 6849, if 1 then discard 9697
# number of discarded genes becomes sensitive to detection limit between 1e-3 and 1e-2 as expected
# consider a gene to be poorly measured if the fraction of counts below limit of detection exceeds a certain threshold
# if this threshold = 0.5, corresponds to requiring the median to be greater than the limit of detection, i.e. detecting the gene in at least half the samples
#for j in range(gene_atb.shape[1]):
#    vls = gene_atb.matrix[:,j][gene_atb.matrix[:,j] > 0]
#    print('\t'.join(['{0:1.3g}'.format(x) for x in np.percentile(vls, [0, 1, 25, 50, 75, 99, 100])]), flush=True)
print('discarding genes with 0.5 samples below detection limit...', flush=True)
detectionlimit = 0.001
fractionsamplesbelowdetection = (gene_atb.matrix < detectionlimit).sum(1)/gene_atb.shape[1]
maxfractionbelowdetection = 0.5
tobediscarded = fractionsamplesbelowdetection > maxfractionbelowdetection
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)
print('fraction of zeros: {0:1.3g}'.format((gene_atb.matrix == 0).sum()/gene_atb.size), flush=True)
print('fraction below detection: {0:1.3g}'.format((gene_atb.matrix < detectionlimit).sum()/gene_atb.size), flush=True)
# if threshold = 0.05 of samples below dection limit
# if detection limit = 1e-30 then discard 2478 genes, if 1e-9 then discard 2484, if 1e-4 then discard 2499, if 1e-3 then discard 2504, if 0.01 then discard 2825, if 0.1 then discard 4529, if 1 then discard 7587
# again, number of discarded genes becomes sensitive to detection limit between 1e-3 and 1e-2 as expected
print('discarding genes with 0.05 samples below detection limit...', flush=True)
detectionlimit = 1e-3
fractionsamplesbelowdetection = (gene_atb.matrix < detectionlimit).sum(1)/gene_atb.shape[1]
maxfractionbelowdetection = 0.05
tobediscarded = fractionsamplesbelowdetection > maxfractionbelowdetection
print(tobediscarded.sum(), flush=True)
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)
print('fraction of zeros: {0:1.3g}'.format((gene_atb.matrix == 0).sum()/gene_atb.size), flush=True)
print('fraction below detection: {0:1.3g}'.format((gene_atb.matrix < detectionlimit).sum()/gene_atb.size), flush=True)
#print('filling in remaining zeros as 0.5*(sample min)...', flush=True)
#nonzeromins = np.zeros(gene_atb.shape[1], dtype='float64')
#for j in range(gene_atb.shape[1]):
#    nonzeromins[j] = gene_atb.matrix[gene_atb.matrix[:,j]>0,j].min()
#    gene_atb.matrix[gene_atb.matrix[:,j]==0,j] = nonzeromins[j]/2.0
print('filling in values below detection as 0.5*detectionlimit...', flush=True)
gene_atb.matrix[gene_atb.matrix < detectionlimit] = detectionlimit/2.0
print(gene_atb, flush=True)

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50); plt.savefig('../../original_data/hugolo_symlnk/hist_5cols_fpkm.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10); plt.savefig('../../original_data/hugolo_symlnk/hist_5rows_fpkm_zscored.png', transparent=True, pad_inches=0, dpi=300)

# log2
print('applying log2 transformation...', flush=True)
gene_atb.matrix = np.log2(gene_atb.matrix)
gene_atb.matrixname += '_log2'

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50); plt.savefig('../../original_data/hugolo_symlnk/hist_5cols_fpkm_log2.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10); plt.savefig('../../original_data/hugolo_symlnk/hist_5rows_fpkm_log2_zscored.png', transparent=True, pad_inches=0, dpi=300)

# normalize samples
#print('applying median shift normalization...', flush=True)
# want to eliminate global fold changes (after log2, global shifts) between samples
# use median sample as reference sample
# compute global shifts relative to reference sample
# subtract global shifts
# note, this corresponds to dividing counts by a sample specific scaling factors
# scaling factors would be almost equivalent to average of sample 50th and 75th %iles
# i.e. if sf = (np.percentile(gene_atb.matrix, 75, 0) + np.percentile(gene_atb.matrix, 50, 0))/2
# then median_shift_from_median  ~ sf - np.median(sf)
# this seems like a better appraoch because it avoids scaling to an arbitrary percentile
median_sample = np.median(gene_atb.matrix, 1)
median_shift_from_median = np.median(gene_atb.matrix - median_sample.reshape(-1,1), 0)
print(median_shift_from_median, flush=True)
#gene_atb.matrix -=  median_shift_from_median.reshape(1,-1)
#gene_atb.rowmeta['median_sample_ref'] = median_sample.copy()
#gene_atb.matrixname += '_medianshift'

# distributions
#plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50); plt.savefig('../../original_data/hugolo_symlnk/hist_5cols_fpkm_log2_medianshift.png', transparent=True, pad_inches=0, dpi=300)
#plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10); plt.savefig('../../original_data/hugolo_symlnk/hist_5rows_fpkm_log2_medianshift_zscored.png', transparent=True, pad_inches=0, dpi=300)

# discard flat genes
print('discarding flat genes...', flush=True)
tobediscarded = 2*gene_atb.matrix.std(1) < np.log2(1.5) # 95% of samples are between (2/3)*mean and (3/2)*mean
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)

# merge duplicate rows
print('merging duplicate rows...', flush=True)
gene_atb.merge(0)
print(gene_atb, flush=True)

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
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50); plt.savefig('../../original_data/hugolo_symlnk/hist_5cols_fpkm_log2_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(gene_atb.matrix[:5,:].T, 10); plt.savefig('../../original_data/hugolo_symlnk/hist_5rows_fpkm_log2_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(gene_atb.matrix.reshape(-1), 1000); plt.savefig('../../original_data/hugolo_symlnk/hist_all_fpkm_log2_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(column_mean.reshape(-1), 10); plt.savefig('../../original_data/hugolo_symlnk/hist_colmeans_fpkm_log2_zscored.png', transparent=True, pad_inches=0, dpi=300)
plt.figure(); plt.hist(column_stdv.reshape(-1), 10); plt.savefig('../../original_data/hugolo_symlnk/hist_colstdvs_fpkm_log2_zscored.png', transparent=True, pad_inches=0, dpi=300)

# save the data
print('saving prepared data...', flush=True)
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/gene_patient_hugolo_rnaseq_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/gene_patient_hugolo_rnaseq_prepared.txt.gz', gene_atb)
with open('../../original_data/hugolo_symlnk/prepare_specs_and_transform_vars.pickle', 'wb') as fw:
    pickle.dump({'median_sample':median_sample, 'row_mean':row_mean, 'row_stdv':row_stdv, 'column_mean':column_mean, 'column_stdv':column_stdv}, fw)
savefolder = '../../input_data/hugolo'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/hugolo_symlnk/gene_patient_hugolo_rnaseq_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/hugolo_symlnk/gene_patient_hugolo_rnaseq_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

# save transpose
print('saving transposed data...', flush=True)
atb_gene = gene_atb.totranspose()
atb_gene.matrixname += '_transposed'
datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared.pickle', atb_gene)
datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared.txt.gz', atb_gene)
savefolder = '../../input_data/hugolo_transposed'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, atb_gene)
shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
