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



# propagate associations up the phenotype tree
print('propagating associations up the phenotype tree...', flush=True)
with open('../../original_data/impc/child_parent_mouse-phenotype-ids_dict.pickle', 'rb') as fr:
    child_parent = pickle.load(fr)

unique_mp_ids = set()
for child, parents in child_parent.items():
    unique_mp_ids.add(child)
    unique_mp_ids.update(parents)

unique_mp_ids.update(gene_atb.columnlabels.tolist())
unique_mp_ids = np.array(list(unique_mp_ids), dtype='object')
mp_names = np.array([mpid_name[mpid] if mpid in mpid_name else 'nan' for mpid in unique_mp_ids], dtype='object')
print('{0!s} unique phenotype ids'.format(unique_mp_ids.size), flush=True)
print('missing phenotype names for {0!s} phenotype ids'.format((mp_names == 'nan').sum()), flush=True)

gene_atb = gene_atb.tolabels(columnlabels=unique_mp_ids.copy())
gene_atb.columnmeta['mp_name'] = mp_names.copy()
gene_atb.matrix = gene_atb.matrix.astype('bool')
gene_atb.updatedtypeattribute()
print(gene_atb, flush=True)

missingchildren = []
for child, parents in child_parent.items():
    newparents = parents.intersection(gene_atb.columnlabels)
    if child not in gene_atb.columnlabels or len(newparents) == 0:
        missingchildren.append(child)
    else:
        child_parent[child] = newparents

print('num missingchildren: {0!s}'.format(len(missingchildren)), flush=True)
for child in missingchildren:
    del child_parent[child]

converged = False
iteration = 0
prevmat = gene_atb.matrix.copy()
print('starting iterations...', flush=True)
print('num associations: {0!s}'.format(gene_atb.matrix.sum()), flush=True)
while not converged:
    for child, parents in child_parent.items():
        c = gene_atb.matrix[:,gene_atb.columnlabels == child] # logical indexing returns 2D array, so the next line will broadcast properly
        gene_atb.matrix[:,np.in1d(gene_atb.columnlabels, list(parents))] += c # performs an OR because gene_atb is boolean
    converged = (gene_atb.matrix == prevmat).all()
    prevmat = gene_atb.matrix.copy()
    iteration += 1
    print('finished iteration {0!s}'.format(iteration), flush=True)
    print('num associations: {0!s}'.format(gene_atb.matrix.sum()), flush=True)

gene_atb.discard((gene_atb.matrix == 0).all(0), 1)
gene_atb.matrix = gene_atb.matrix.astype('float64')
gene_atb.updatedtypeattribute()
gene_atb.matrixname += '_propagated'
print(gene_atb, flush=True)



# save the data
print('saving prepared data...', flush=True)
gene_atb.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_propagated_prepared.pickle', gene_atb)
datasetIO.save_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_propagated_prepared.txt.gz', gene_atb)
savefolder = '../../input_data/impc_binary_propagated'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, gene_atb)
shutil.copyfile('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_propagated_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_propagated_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
