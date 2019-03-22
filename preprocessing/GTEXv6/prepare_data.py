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
from matplotlib import pyplot as plt

# load the data
gene_atb = datasetIO.load_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_counts.pickle')

# scale counts
gene_atb.matrix = np.exp(np.log(gene_atb.matrix) - np.log(gene_atb.columnmeta['auc'].reshape(1,-1)) + (np.log(4) + 7*np.log(10)))
gene_atb.matrixname += '_scaledcounts'
datasetIO.save_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_scaledcounts.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_scaledcounts.txt.gz', gene_atb)

# shuffle the data
gene_atb.reorder(np.random.permutation(gene_atb.shape[0]), 0)
gene_atb.reorder(np.random.permutation(gene_atb.shape[1]), 1)
print(gene_atb)

# strip version from ensembl_gene_ids
gene_atb.rowlabels = np.array([x.rsplit('.', maxsplit=1)[0] for x in gene_atb.rowlabels], dtype='object')

# add hgnc metadata
hgncmetadata = mapper.annotate_genes(field='ensembl_gene_id', values=gene_atb.rowlabels, metadatapath='../../mappings/hgnc/hgnc_20181016_complete_set.txt', drop_duplicates=True)
gene_atb.rowmeta.update(hgncmetadata)
gene_atb.rowname = 'ensembl_gene_id'
gene_atb.rowlabels = gene_atb.rowmeta['ensembl_gene_id'].copy()
del gene_atb.rowmeta['ensembl_gene_id']
print(gene_atb.rowmeta.keys())
tobediscarded = np.logical_or(gene_atb.rowmeta['entrez_id'] == 'nan', gene_atb.rowmeta['symbol'] == 'nan')
gene_atb.discard(tobediscarded, 0)
print(gene_atb)

# discard pseudogenes
print(np.unique(gene_atb.rowmeta['locus_type']).tolist())
tobediscarded = ~np.in1d(gene_atb.rowmeta['locus_type'], ['RNA, long non-coding', 'RNA, micro', 'T cell receptor gene', 'gene with protein product', 'immunoglobulin gene', 'protocadherin'])
gene_atb.discard(tobediscarded, 0)
print(gene_atb)

# handle zeros
# counts are scaled to a library size (mapped reads) of 4e7, which is a typical value (median mapped reads for gtex is 7e7)
# so a scaled count of 1 represents the limit of detection
# consider a gene to be poorly measured if the fraction of counts below 1 (below limit of detection) exceeds a certain threshold
# if this threshold = 0.5, corresponds to requiring the median to be greater than the limit of detection
# i.e. tobediscarded = np.median(gene_atb.matrix, 1) <= 1
detectionlimit = 1
fractionsamplesbelowdetection = (gene_atb.matrix < detectionlimit).sum(1)/gene_atb.shape[1]
maxfractionbelowdetection = 0.5
tobediscarded = fractionsamplesbelowdetection > maxfractionbelowdetection
gene_atb.discard(tobediscarded, 0)
print(gene_atb)
print('fraction of zeros: {0:1.3g}'.format((gene_atb.matrix == 0).sum()/gene_atb.size))
detectionlimit = 1
fractionsamplesbelowdetection = (gene_atb.matrix < detectionlimit).sum(1)/gene_atb.shape[1]
maxfractionbelowdetection = 0.05
u_specific_tissues = np.unique(gene_atb.columnmeta['specific_tissue'])
specific_tissue_mins = np.zeros((gene_atb.shape[0], u_specific_tissues.size), dtype='float64')
for j, specific_tissue in enumerate(u_specific_tissues):
    specific_tissue_mins[:,j] = np.min(gene_atb.matrix[:,gene_atb.columnmeta['specific_tissue'] == specific_tissue], 1)
robustlimit = 10
nospecifictissuesrobust = (specific_tissue_mins < robustlimit).all(1)
tobediscarded = np.logical_and(fractionsamplesbelowdetection > maxfractionbelowdetection, nospecifictissuesrobust)
gene_atb.discard(tobediscarded, 0)
print(gene_atb)
print('fraction of zeros: {0:1.3g}'.format((gene_atb.matrix == 0).sum()/gene_atb.size))
nonzeromins = np.zeros(gene_atb.shape[1], dtype='float64')
for j in range(gene_atb.shape[1]):
    nonzeromins[j] = gene_atb.matrix[gene_atb.matrix[:,j]>0,j].min()
    gene_atb.matrix[gene_atb.matrix[:,j]==0,j] = nonzeromins[j]/2.0
print(gene_atb)

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)

# log2
gene_atb.matrix = np.log2(gene_atb.matrix)

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)

# normalize samples
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
gene_atb.matrix -=  median_shift_from_median.reshape(1,-1)
gene_atb.rowmeta['median_sample_ref'] = median_sample.copy()

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)

# discard flat genes
tobediscarded = 2*gene_atb.matrix.std(1) < np.log2(1.5) # 95% of samples are between (2/3)*mean and (3/2)*mean
gene_atb.discard(tobediscarded, 0)
print(gene_atb)

# merge duplicate rows
gene_atb.merge(0)
print(gene_atb)

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50)
plt.figure(); plt.hist(((gene_atb.matrix[:5,:] - gene_atb.matrix[:5,:].mean(1, keepdims=True))/gene_atb.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)

# annotate the genes
gene_tis = copy.deepcopy(gene_atb)
gene_tis.columnlabels = gene_tis.columnmeta['general_tissue'].copy()
gene_tis.columnname = 'general_tissue'
gene_tis.merge(1)
sorted_rows = np.sort(gene_tis.matrix, 1)
row_min = sorted_rows[:,0]
row_second_lowest = sorted_rows[:,1]
row_max = sorted_rows[:,-1]
row_second_highest = sorted_rows[:,-2]
row_median = np.median(gene_tis.matrix, 1)
is_overexpressed_and_specific = np.logical_and(row_max - row_median >= np.log2(5), row_max - row_second_highest >= np.log2(2.5))
gene_atb.rowmeta['5Xoverexpressed_and_2.5Xspecific_tissue'] = gene_tis.columnlabels[np.argmax(gene_tis.matrix, 1)]
gene_atb.rowmeta['5Xoverexpressed_and_2.5Xspecific_tissue'][~is_overexpressed_and_specific] = 'none'
is_underexpressed_and_specific = np.logical_and(row_median - row_min >= np.log2(5), row_second_lowest - row_min >= np.log2(2.5))
gene_atb.rowmeta['5Xunderexpressed_and_2.5Xspecific_tissue'] = gene_tis.columnlabels[np.argmin(gene_tis.matrix, 1)]
gene_atb.rowmeta['5Xunderexpressed_and_2.5Xspecific_tissue'][~is_underexpressed_and_specific] = 'none'
del gene_tis, sorted_rows, row_min, row_second_lowest, row_max, row_second_highest, row_median, is_overexpressed_and_specific, is_underexpressed_and_specific

# standardize the data
row_mean = gene_atb.matrix.mean(1, keepdims=True)
row_stdv = gene_atb.matrix.std(1, keepdims=True)
gene_atb.rowmeta['row_mean_ref'] = row_mean.copy()
gene_atb.rowmeta['row_stdv_ref'] = row_stdv.copy()
standardized_row_mean = (row_mean - row_mean.mean())/row_mean.std()
standardized_row_stdv = (row_stdv - row_stdv.mean())/row_stdv.std()
gene_atb.matrix = (gene_atb.matrix - row_mean)/row_stdv
column_mean = gene_atb.matrix.mean(0)
column_stdv = gene_atb.matrix.std(0)
adjusted_row_mean = standardized_row_mean*column_stdv.mean() + column_mean.mean()
adjusted_row_stdv = standardized_row_stdv*column_stdv.mean() + column_mean.mean()
gene_atb.matrix = np.concatenate((gene_atb.matrix, adjusted_row_mean, adjusted_row_stdv), 1)
gene_atb.columnlabels = np.append(gene_atb.columnlabels, np.array(['adjusted_row_mean', 'adjusted_row_stdv'], dtype='object'))
for label in gene_atb.columnmeta:
    if gene_atb.columnmeta[label].dtype == 'object':
        gene_atb.columnmeta[label] = np.append(gene_atb.columnmeta[label], np.array(['-666', '-666'], dtype='object'))
    else:
        gene_atb.columnmeta[label] = np.append(gene_atb.columnmeta[label], np.array([np.nan, np.nan], dtype=gene_atb.columnmeta[label].dtype))
gene_atb.updateshapeattribute()
gene_atb.updatesizeattribute()
column_mean = gene_atb.matrix.mean(0, keepdims=True)
column_stdv = gene_atb.matrix.std(0, keepdims=True)
#gene_atb.matrix = (gene_atb.matrix - column_mean)/column_stdv
print(gene_atb)

# distributions
plt.figure(); plt.hist(gene_atb.matrix[:,:5], 50)
plt.figure(); plt.hist(gene_atb.matrix[:5,:].T, 10)
plt.figure(); plt.hist(gene_atb.matrix.reshape(-1), 1000)
plt.figure(); plt.hist(column_mean.reshape(-1), 10)
plt.figure(); plt.hist(column_stdv.reshape(-1), 10)

# save the data
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_scaledcounts_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_scaledcounts_prepared.txt.gz', gene_atb)
with open('../../original_data/GTEXv6/prepare_specs_and_transform_vars.pickle', 'wb') as fw:
    pickle.dump({'median_sample':median_sample, 'row_mean':row_mean, 'row_stdv':row_stdv, 'standardized_row_mean':standardized_row_mean, 'standardized_row_stdv':standardized_row_stdv, 'adjusted_row_mean':adjusted_row_mean, 'adjusted_row_stdv':adjusted_row_stdv, 'column_mean':column_mean, 'column_stdv':column_stdv}, fw)
savefolder = '../../input_data/GTEXv6'
os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_scaledcounts_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))
