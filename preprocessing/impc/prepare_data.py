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

# load the data
print('loading data...', flush=True)
print('values are p-values with non-significant associations (pvalue > 1e-4) imputed with pvalue=1', flush=True)
gene_atb = datasetIO.load_datamatrix('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed.csv.gz', delimiter=',', getmetadata=False) # (3455, 295)
gene_atb.rowname = 'mgd_id'
gene_atb.columnname = 'mp_id'
gene_atb.matrixname = 'gene_phenotype_associations_from_impc'

# rescale data
print('rescaling data...', flush=True)
print('pvalues have a strange distribution due to thresholding of high and low pvalues.', flush=True)
print('convert pvalues to -log10(p) and re-scale so median has a value of 0.5 and clip the values that are greater than 1.', flush=True)
nlpcutoff = 2*(-np.log10(np.percentile(gene_atb.matrix[gene_atb.matrix < 1], 50))) # 10.96
pvalcutoff = 10**(-nlpcutoff) # 1e-11
gene_atb.matrix[gene_atb.matrix < pvalcutoff] = pvalcutoff
gene_atb.matrix = -np.log10(gene_atb.matrix)/nlpcutoff
gene_atb.matrix[gene_atb.matrix >= 1] = 1
gene_atb.matrix[gene_atb.matrix <= 0] = 0
gene_atb.matrixname += '_rescaled'
print('min: {0!s}, median: {1!s}, max: {2!s}'.format(gene_atb.matrix.min(), np.median(gene_atb.matrix[gene_atb.matrix > 0]), gene_atb.matrix.max()), flush=True)

# save rescaled data
print('saving rescaled data...', flush=True)
datasetIO.save_datamatrix('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed_rescaled.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed_rescaled.txt.gz', gene_atb)

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

# save the data
print('saving prepared data...', flush=True)
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_rescaled_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_rescaled_prepared.txt.gz', gene_atb)
savefolder = '../../input_data/impc'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/impc/gene_phenotype_impc_trimmed_rescaled_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/impc/gene_phenotype_impc_trimmed_rescaled_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
