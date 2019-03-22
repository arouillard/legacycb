# -*- coding: utf-8 -*-
"""
Clinical Outcome Classifier
get scores for individual features
multiple hypothesis correction by dataset or altogether?
what is the best test?
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
import machinelearning.dataclasses as dc
import pickle

# load clusters
with open('clusters.pickle', 'rb') as fr:
    gene_syms, gene_ids, cluster_ids = pickle.load(fr)
unique_cluster_ids = np.array([str(x) for x in np.unique(cluster_ids)], dtype='object')

# create matrix
gene_clust = dc.datamatrix(rowname='gene_sym',
                           rowlabels=gene_syms,
                           rowmeta={'gene_id':gene_ids},
                           columnname='cluster_id',
                           columnlabels=unique_cluster_ids,
                           columnmeta={},
                           matrixname='gene_cluster_assignments_from_denoising_autoencoder_applied_to_GTEX',
                           matrix=np.zeros((gene_syms.size, unique_cluster_ids.size), dtype='bool'))
for j, cluster_id in enumerate(gene_clust.columnlabels):
    gene_clust.matrix[:,j] = cluster_ids == int(cluster_id)

# write matrix
with open('gene_cluster_matrix.pickle', 'wb') as fw:
    pickle.dump(gene_clust, fw)
