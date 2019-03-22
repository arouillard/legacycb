# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import datasetIO
import mapper
import os
import shutil
import pickle

# load the data
print('loading data...', flush=True)
print('values are p-values with non-significant associations (pvalue > 1e-4) imputed with pvalue=1', flush=True)
gene_atb = datasetIO.load_datamatrix('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed.csv.gz', delimiter=',', getmetadata=False) # (3455, 295)
gene_atb.rowname = 'mgd_id'
gene_atb.columnname = 'mp_id'
gene_atb.matrixname = 'gene_phenotype_associations_from_impc'

# threshold the data
print('thresholding data...', flush=True)
print('because significant associations have p-value 1e-4 or less, perhaps relative p-values are not informative and better to threshold', flush=True)
gene_atb.matrix = np.float64(gene_atb.matrix < 1)
gene_atb.matrixname += '_thresholded'
print('matrix sparsity: {0!s}, row median sparsity: {1!s}, column median sparsity: {2!s}'.format(gene_atb.matrix.sum()/gene_atb.size, np.median(gene_atb.matrix.sum(1)/gene_atb.shape[1]), np.median(gene_atb.matrix.sum(0)/gene_atb.shape[0])), flush=True)

# save thresholded data
print('saving thresholded data...', flush=True)
datasetIO.save_datamatrix('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed_thresholded.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed_thresholded.txt.gz', gene_atb)

# shuffle the data
print('shuffling data...', flush=True)
gene_atb.reorder(np.random.permutation(gene_atb.shape[0]), 0)
gene_atb.reorder(np.random.permutation(gene_atb.shape[1]), 1)
print(gene_atb, flush=True)

# add hgnc metadata
print('adding hgnc metadata data...', flush=True)
hgncmetadata = mapper.annotate_genes(field='mgd_id', values=gene_atb.rowlabels, metadatapath='../../mappings/hgnc/hgnc_20181016_complete_set.txt', drop_duplicates=True)
gene_atb.rowmeta.update(hgncmetadata)
gene_atb.rowname = 'ensembl_gene_id'
gene_atb.rowlabels = gene_atb.rowmeta['ensembl_gene_id'].copy()
del gene_atb.rowmeta['ensembl_gene_id']
print(gene_atb.rowmeta.keys(), flush=True)
tobediscarded = np.logical_or.reduce((gene_atb.rowlabels == 'nan', gene_atb.rowmeta['entrez_id'] == 'nan', gene_atb.rowmeta['symbol'] == 'nan'))
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)

# discard pseudogenes
print('discarding pseudogenes data...', flush=True)
print(np.unique(gene_atb.rowmeta['locus_type']).tolist())
tobediscarded = ~np.in1d(gene_atb.rowmeta['locus_type'], ['RNA, long non-coding', 'RNA, micro', 'T cell receptor gene', 'gene with protein product', 'immunoglobulin gene', 'protocadherin'])
gene_atb.discard(tobediscarded, 0)
print(gene_atb, flush=True)

# add mp metadata
print('adding mouse phenotype metadata data...', flush=True)
with open('../../original_data/impc/mpid_name_dict.pickle', 'rb') as fr:
    mpid_name = pickle.load(fr)
gene_atb.columnmeta['mp_name'] = np.array([mpid_name[mpid] if mpid in mpid_name else 'nan' for mpid in gene_atb.columnlabels], dtype='object')
print('missing phenotype names for {0!s} phenotype ids'.format((gene_atb.columnmeta['mp_name'] == 'nan').sum()), flush=True)

# save the data
print('saving prepared data...', flush=True)
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_prepared.txt.gz', gene_atb)
savefolder = '../../input_data/impc_binary'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
