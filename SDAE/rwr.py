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

import os
#import gzip
import numpy as np
import copy
from machinelearning import datasetselection, featureselection
from machinelearning import dataclasses as dc
from machinelearning import datasetIO
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt

distance_sigma = 2*0.0011 # 0.05 # 0.0011 is mean of min gene-gene distance
restart_probability = 0.5
atb_minscore = {"Parkinson's Disease":4, "Alzheimer's Disease":4, 'TNT39':0, 'TNT7':0}
score_flattening = 1000
tol = 1e-5 # 1e-6
itr = 100

# analysis version
results_folder = 'rwr_v1'
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

gene_proj = datasetIO.load_datamatrix('gene_atb_matrix_2d_dnn_projection.pickle')

gene_gene = datasetIO.load_datamatrix('gene_gene_matrix_euclidean_distance_from_projection.pickle')
gene_gene.matrix = np.exp(-1*gene_gene.matrix**2/2/distance_sigma**2)

#datasetinfo = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_nonspecific', 'name':'Tissue Over-expression 6-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_nonspecific.pickle'}
#datasetinfo = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_specific', 'name':'Specific Tissue Over-expression 6-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_3xspecific.pickle'}
datasetinfo = {'folder':'gtextissue', 'abbreviation':'pankaj_disease_genes', 'name':'Ranked Disease Genes from Pankaj', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_disease_matrix_pankaj.pickle'}
gene_atb = datasetIO.load_datamatrix(datasetinfo['path'])
for j, atb in enumerate(gene_atb.columnlabels):
    gene_atb.matrix[gene_atb.matrix[:,j] < atb_minscore[atb],j] = 0
gene_atb.matrix = gene_atb.matrix**(1/score_flattening) #################
gene_atb.updatedtypeattribute()

#union_genes = np.union1d(gene_proj.rowlabels, np.union1d(gene_gene.columnlabels, gene_atb.rowlabels))
union_genes = gene_gene.columnlabels.copy()

gene_proj = gene_proj.tolabels(rowlabels=union_genes.copy(), fillvalue=np.nan)
is_nan_gene = np.isnan(gene_proj.matrix).all(1)

gene_gene = gene_gene.tolabels(rowlabels=union_genes.copy(), columnlabels=union_genes.copy())
gene_gene.matrix[np.eye(gene_gene.shape[0], dtype='bool')] = 1
gene_gene.matrix /= gene_gene.matrix.sum(0, keepdims=True)

gene_atb = gene_atb.tolabels(rowlabels=union_genes.copy())
gene_atb.matrix /= gene_atb.matrix.sum(0, keepdims=True)

#gene_atb_prop = copy.deepcopy(gene_atb)
#err = 1
#ctr = 0
#while err > tol and ctr < itr:
#    ctr += 1
#    print(ctr)
#    prev_mat = gene_atb_prop.matrix.copy()
#    gene_atb_prop.matrix = (1 - restart_probability)*gene_gene.matrix.dot(gene_atb_prop.matrix) + restart_probability*gene_atb_prop.matrix
#    err = np.mean(np.abs(prev_mat - gene_atb_prop.matrix))

numbootstraps = 30
propmat = np.zeros(gene_atb.shape, dtype='float64')
gene_atb_prop = copy.deepcopy(gene_atb)
for bootstrap in range(numbootstraps):
    gene_atb_prop.matrix[:] = 0
    for j in range(gene_atb.shape[1]):
        hidx = (gene_atb.matrix[:,j] > 0).nonzero()[0]
        bidx = np.random.choice(hidx, hidx.size, replace=True)
        u_bidx, c_bidx = np.unique(bidx, return_counts=True)
        gene_atb_prop.matrix[u_bidx,j] = c_bidx # would need to modify to account for weighted seeds
    gene_atb_prop.matrix /= gene_atb_prop.matrix.sum(0, keepdims=True)
    err = 1
    ctr = 0
    while err > tol and ctr < itr:
        ctr += 1
        print(ctr)
        prev_mat = gene_atb_prop.matrix.copy()
        gene_atb_prop.matrix = (1 - restart_probability)*gene_gene.matrix.dot(gene_atb_prop.matrix) + restart_probability*gene_atb_prop.matrix
        err = np.mean(np.abs(prev_mat - gene_atb_prop.matrix))
    propmat += gene_atb_prop.matrix/numbootstraps
gene_atb_prop.matrix = propmat
del propmat

gene_atb_prop.cluster(1, metric='cosine', method='average')
gene_atb = gene_atb.tolabels(columnlabels=gene_atb_prop.columnlabels.copy())

rows = 4
cols = 8
yub = 0.5
ylb = -0.35
xub = 0.45
xlb = -0.4
fg, axs = plt.subplots(rows, cols, figsize=(10,5))
fg.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.9, wspace=0.2, hspace=0.2)
fg.suptitle('{0}, {1!s} features'.format(datasetinfo['name'], gene_atb.shape[1]), fontsize=8)
axs = axs.reshape(-1)
num_axs = rows*cols
for i, (feature, prop_values, orig_values) in enumerate(zip(gene_atb_prop.columnlabels, gene_atb_prop.matrix.T, gene_atb.matrix.T)):
    if i >= num_axs:
        break
    ax = axs[i]
    is_seed = orig_values > 0
#    cub = prop_values[is_seed].min()
    cub = np.percentile(prop_values[~is_seed], 99)
#    cub = prop_values[~is_seed].max()
    clb = prop_values.min()
    ax.scatter(gene_proj.matrix[is_seed,0], gene_proj.matrix[is_seed,1], s=0.1, c=prop_values[is_seed], marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), vmin=clb, vmax=cub, zorder=1)
    ax.scatter(gene_proj.matrix[~is_seed,0], gene_proj.matrix[~is_seed,1], s=0.1, c=prop_values[~is_seed], marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), vmin=clb, vmax=cub, zorder=0)
#    ax.scatter(gene_proj.matrix[:,0], gene_proj.matrix[:,1], s=0.1, c=prop_values, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), vmin=clb, vmax=cub, zorder=0)
    ax.set_title(feature[:35], fontsize=4)
    ax.set_ylim(ylb, yub)
    ax.set_xlim(xlb, xub)
    ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
    ax.set_frame_on(False)
    ax.axvline(xlb, linewidth=0.5, color='k')
    ax.axvline(xub, linewidth=0.5, color='k')
    ax.axhline(ylb, linewidth=0.5, color='k')
    ax.axhline(yub, linewidth=0.5, color='k')
if i < num_axs:
    for j in range(i+1, num_axs):
        plt.delaxes(axs[j])
fg.savefig('{0}/{1}_maps_ds{2!s}_rp{3!s}_sf{4!s}.png'.format(results_folder, datasetinfo['abbreviation'], distance_sigma, restart_probability, score_flattening), transparent=True, pad_inches=0, dpi=1200)
plt.close()
