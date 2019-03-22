# -*- coding: utf-8 -*-
"""
Clinical Outcome Classifier
get scores for individual features
multiple hypothesis correction by dataset or altogether?
what is the best test?
"""

import sys
#custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
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
import matplotlib.pyplot as plt

# load distance matrices
with open('gene_gene_matrix_euclidean_distance_from_projection.pickle', 'rb') as fr:
    gene_gene_prj = pickle.load(fr)
assert (gene_gene_prj.rowlabels == gene_gene_prj.columnlabels).all()
triu_mask = np.triu(np.ones(gene_gene_prj.shape, dtype='bool'), 1)

with open('gene_gene_matrix_angular_cosine_distance_from_zscored_tissue_expression.pickle', 'rb') as fr:
    gene_gene_ang = pickle.load(fr)
assert (gene_gene_ang.rowlabels == gene_gene_ang.columnlabels).all()
assert (gene_gene_prj.rowlabels == gene_gene_ang.rowlabels).all()
gene_gene_ang = gene_gene_ang.matrix[triu_mask]

with open('gene_gene_matrix_euclidean_distance_from_zscored_tissue_expression.pickle', 'rb') as fr:
    gene_gene_euc = pickle.load(fr)
assert (gene_gene_euc.rowlabels == gene_gene_euc.columnlabels).all()
assert (gene_gene_prj.rowlabels == gene_gene_euc.rowlabels).all()
gene_gene_euc = gene_gene_euc.matrix[triu_mask]

gene_gene_prj = gene_gene_prj.matrix[triu_mask]
del triu_mask

ri5 = np.random.choice(gene_gene_prj.size, 100000, replace=False)
ri6 = np.random.choice(gene_gene_prj.size, 1000000, replace=False)

fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.8/6.5, 0.8/6.5, 5.5/6.5, 5.5/6.5])
ax.plot(gene_gene_ang[ri6], gene_gene_prj[ri6], linestyle='None', linewidth=0, marker='o', markerfacecolor='k', markeredgecolor='k', markersize=2, markeredgewidth=0, alpha=0.1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_ylabel('Gene-Gene Similarity from DAE Embedding', fontsize=16)
ax.set_xlabel('Gene-Gene Similarity from Expression', fontsize=16)
ax.tick_params(axis='both', which='major', bottom='on', top='off', labelbottom='on', labeltop='off', left='on', right='off', labelleft='on', labelright='off', pad=4)
fg.savefig('compare_gene_sims.png', transparent=True, pad_inches=0, dpi=600)

