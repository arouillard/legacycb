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
gene_atb = datasetIO.load_datamatrix('../../original_data/phenodigm/geneid_meshid_datamatrix_trimmed.csv.gz', delimiter=',', getmetadata=False)
gene_atb.rowname = 'entrez_id'
gene_atb.columnname = 'mesh_id'
gene_atb.matrixname = 'gene_disease_associations_from_phenodigm-qtq'

# clip the data (there are 22 entries slightly above 1)
# what do the values mean?
# values have a strange distribution. 50% are less than 0.2, 97% are less than 0.5. min value is 0.08. max value is 1.15.
# re-scale so median has a value of 0.5 and clip the values that are greater than 1 (8%)
print('rescaling data...', flush=True)
pct50 = np.percentile(gene_atb.matrix[gene_atb.matrix > 0], 50)
gene_atb.matrix /= 2*pct50
gene_atb.matrix[gene_atb.matrix > 1] = 1
gene_atb.matrixname += '_rescaled'
print('saving rescaled data...', flush=True)
datasetIO.save_datamatrix('../../original_data/phenodigm/gene_disease_phenodigm-qtq_trimmed_rescaled.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/phenodigm/gene_disease_phenodigm-qtq_trimmed_rescaled.txt.gz', gene_atb)

# shuffle the data
print('shuffling data...', flush=True)
gene_atb.reorder(np.random.permutation(gene_atb.shape[0]), 0)
gene_atb.reorder(np.random.permutation(gene_atb.shape[1]), 1)
print(gene_atb)

# add hgnc metadata
print('adding hgnc metadata data...', flush=True)
hgncmetadata = mapper.annotate_genes(field='entrez_id', values=gene_atb.rowlabels, metadatapath='../../mappings/hgnc/hgnc_20181016_complete_set.txt', drop_duplicates=True)
gene_atb.rowmeta.update(hgncmetadata)
gene_atb.rowname = 'ensembl_gene_id'
gene_atb.rowlabels = gene_atb.rowmeta['ensembl_gene_id'].copy()
del gene_atb.rowmeta['ensembl_gene_id']
print(gene_atb.rowmeta.keys())
tobediscarded = np.logical_or.reduce((gene_atb.rowlabels == 'nan', gene_atb.rowmeta['entrez_id'] == 'nan', gene_atb.rowmeta['symbol'] == 'nan'))
gene_atb.discard(tobediscarded, 0)
print(gene_atb)

# discard pseudogenes
print('discarding pseudogenes data...', flush=True)
print(np.unique(gene_atb.rowmeta['locus_type']).tolist())
tobediscarded = ~np.in1d(gene_atb.rowmeta['locus_type'], ['RNA, long non-coding', 'RNA, micro', 'T cell receptor gene', 'gene with protein product', 'immunoglobulin gene', 'protocadherin'])
gene_atb.discard(tobediscarded, 0)
print(gene_atb)

# save the data
print('saving prepared data...', flush=True)
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/phenodigm/gene_disease_phenodigm-qtq_trimmed_rescaled_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/phenodigm/gene_disease_phenodigm-qtq_trimmed_rescaled_prepared.txt.gz', gene_atb)
savefolder = '../../input_data/phenodigm'
os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/phenodigm/gene_disease_phenodigm-qtq_trimmed_rescaled_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
