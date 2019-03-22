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
from machinelearning import datasetselection, featureselection, datasetIO
import machinelearning.dataclasses as dc
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import fastcluster

# load distance matrix
with open('gene_gene_matrix_euclidean_distance_from_projection.pickle', 'rb') as fr:
    gene_gene = pickle.load(fr)

# prefer ward linkage for euclidean distance or at least this case
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'ward')
#plt.figure()
#hierarchy.dendrogram(lnk)
si = hierarchy.leaves_list(lnk).astype('int64')
#cst = hierarchy.cut_tree(lnk, n_clusters=87).reshape(-1)

# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)
if ~(gene_proj.rowlabels == gene_gene.rowlabels).all():
    raise ValueError('genes not aligned')
gene_proj.reorder(si, 0)
#gene_gene.reorder(si, 0)
#gene_gene.reorder(si, 1)
ordered_genes = gene_proj.rowlabels.copy()
del gene_gene, lnk, si

# load  examples
print('loading examples...', flush=True)
with open('blood_targets.median_ratio_blood-to-non-perf_gt10_floor0.1.non-perf_0.9_p90_lt5.gene_list.txt', 'rt') as fr:
    uncleaned_examples = np.array(list(set([x.strip() for x in fr.read().split('\n')])), dtype='object')
    examples = np.array([x.upper().split('.')[0] for x in uncleaned_examples], dtype='object')
gene_proj.rowmeta['is_example'] = np.logical_or.reduce((np.in1d(gene_proj.rowlabels, examples), np.in1d(gene_proj.rowmeta['Ensemble Acc'], examples), np.in1d(gene_proj.rowmeta['GeneID'], examples)))

mapped_examples = uncleaned_examples[np.logical_or.reduce((np.in1d(examples, gene_proj.rowlabels), np.in1d(examples, gene_proj.rowmeta['Ensemble Acc']), np.in1d(examples, gene_proj.rowmeta['GeneID'])))]
unmapped_examples = uncleaned_examples[~np.in1d(uncleaned_examples, mapped_examples)]
with open('temp_highlight_selected_unmapped_examples.txt', 'wt') as fw:
    fw.write('\n'.join(unmapped_examples))

# plot
yub = 0.45
ylb = -0.35
xub = 0.55
xlb = -0.4
fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.25/6.5, 0.25/6.5, 6/6.5, 6/6.5]) # left, bottom, width, height
ax.plot(gene_proj.matrix[gene_proj.rowmeta['is_example'],0], gene_proj.matrix[gene_proj.rowmeta['is_example'],1], 'sr', markersize=3, markeredgewidth=0, alpha=1, zorder=1, label='selected')
ax.plot(gene_proj.matrix[~gene_proj.rowmeta['is_example'],0], gene_proj.matrix[~gene_proj.rowmeta['is_example'],1], 'ok', markersize=3, markeredgewidth=0, alpha=0.01, zorder=0, label='other')
ax.set_ylim(ylb, yub)
ax.set_xlim(xlb, xub)
ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
ax.set_frame_on(False)
ax.legend(loc='lower left', ncol=1, fontsize=8, frameon=False, borderpad=5, labelspacing=0.1, handletextpad=0.1, borderaxespad=0)
fg.savefig('temp_highlight_selected.png', transparent=True, pad_inches=0, dpi=600)

fg, ax = plt.subplots(1, 1, figsize=(3.25,2))
ax.set_position([0.5/3.25, 0.35/2, 2.5/3.25, 1.55/2]) # left, bottom, width, height
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['is_example'],0], gene_proj.matrix[gene_proj.rowmeta['is_example'],1], s=5, c='r', marker='+', alpha=1, edgecolors='r', linewidths=0.5, zorder=1, label='selected')
ax.scatter(gene_proj.matrix[~gene_proj.rowmeta['is_example'],0], gene_proj.matrix[~gene_proj.rowmeta['is_example'],1], s=1, c='k', marker='o', alpha=0.1, edgecolors='k', linewidths=0, zorder=0, label='other')
ax.set_xlabel('SDAE-1', fontsize=8, labelpad=2)
ax.set_ylabel('SDAE-2', fontsize=8, labelpad=4)
ax.tick_params(axis='both', which='major', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off', labelsize=8)
ax.legend(loc='center right', ncol=1, fontsize=8, frameon=False, borderpad=0.1, labelspacing=0, handletextpad=0, borderaxespad=0)
ax.set_ylim(ylb, yub)
ax.set_xlim(xlb, xub)
fg.savefig('temp_highlight_selected_formatted.png', dpi=600, transparent=True, pad_inches=0)

fg, ax = plt.subplots(1, 1, figsize=(9.75,6))
ax.set_position([1.5/9.75, 1.05/6, 7.5/9.75, 4.7/6]) # left, bottom, width, height
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['is_example'],0], gene_proj.matrix[gene_proj.rowmeta['is_example'],1], s=20, c='r', marker='+', alpha=1, edgecolors='r', linewidths=1.5, zorder=1, label='selected')
ax.scatter(gene_proj.matrix[~gene_proj.rowmeta['is_example'],0], gene_proj.matrix[~gene_proj.rowmeta['is_example'],1], s=1, c='k', marker='o', alpha=0.1, edgecolors='k', linewidths=0, zorder=0, label='other')
ax.set_xlabel('SDAE-1', fontsize=16, labelpad=2)
ax.set_ylabel('SDAE-2', fontsize=16, labelpad=4)
ax.tick_params(axis='both', which='major', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off', labelsize=16)
ax.legend(loc='upper left', ncol=1, fontsize=16, frameon=False, borderpad=0.1, labelspacing=0, handletextpad=0, borderaxespad=0)
ax.set_ylim(ylb, yub)
ax.set_xlim(xlb, xub)
fg.savefig('temp_highlight_selected_formatted_big.png', dpi=600, transparent=True, pad_inches=0)

yub = 0.0
ylb = -0.35
xub = 0.35
xlb = 0.0
fg, ax = plt.subplots(1, 1, figsize=(9.75,6))
ax.set_position([1.5/9.75, 1.05/6, 7.5/9.75, 4.7/6]) # left, bottom, width, height
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['is_example'],0], gene_proj.matrix[gene_proj.rowmeta['is_example'],1], s=20, c='r', marker='+', alpha=1, edgecolors='r', linewidths=1.5, zorder=1, label='selected')
ax.scatter(gene_proj.matrix[~gene_proj.rowmeta['is_example'],0], gene_proj.matrix[~gene_proj.rowmeta['is_example'],1], s=2, c='k', marker='o', alpha=0.5, edgecolors='k', linewidths=0, zorder=0, label='other')
ax.set_xlabel('SDAE-1', fontsize=16, labelpad=2)
ax.set_ylabel('SDAE-2', fontsize=16, labelpad=4)
ax.tick_params(axis='both', which='major', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off', labelsize=16)
ax.legend(loc='upper right', ncol=1, fontsize=16, frameon=False, borderpad=0.1, labelspacing=0, handletextpad=0, borderaxespad=0)
ax.set_ylim(ylb, yub)
ax.set_xlim(xlb, xub)
fg.savefig('temp_highlight_selected_formatted_big_zoom.png', dpi=600, transparent=True, pad_inches=0)

cluster_bounds = {0:{'x':[-1.0, 1.0], 'y':[-1.0, 1.0]},
                  1:{'x':[0.0457, 0.0669], 'y':[-0.1962, -0.1597]},
                  2:{'x':[0.1652, 0.1918], 'y':[-0.2775, -0.2329]},
                  3:{'x':[0.1822, 0.2348], 'y':[-0.1682, -0.1254]},
                  4:{'x':[0.2117, 0.2273], 'y':[-0.1967, -0.1683]},
                  5:{'x':[0.2349, 0.2851], 'y':[-0.1628, -0.1344]}}

hitmat = np.zeros((gene_proj.shape[0], len(cluster_bounds)), dtype='bool')
for cluster in range(len(cluster_bounds)):
    bounds = cluster_bounds[cluster]
    hitmat[:,cluster] = np.all(np.concatenate((gene_proj.rowmeta['is_example'].reshape(-1,1), (gene_proj.matrix[:,0] > bounds['x'][0]).reshape(-1,1), (gene_proj.matrix[:,0] < bounds['x'][1]).reshape(-1,1), (gene_proj.matrix[:,1] > bounds['y'][0]).reshape(-1,1), (gene_proj.matrix[:,1] < bounds['y'][1]).reshape(-1,1)), 1), 1)

gene_cluster = copy.deepcopy(gene_proj)
gene_cluster.columnname = 'cluster'
gene_cluster.columnlabels = np.array(['Cluster{0!s}'.format(x) for x in range(len(cluster_bounds))], dtype='object')
gene_cluster.columnmeta = {}
gene_cluster.matrixname = 'selected_clusters'
gene_cluster.matrix = hitmat.astype('float64')
gene_cluster.updatedtypeattribute()
gene_cluster.updatesizeattribute()
gene_cluster.updateshapeattribute()
datasetIO.save_datamatrix('C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_cluster_matrix_sarah.pickle', gene_cluster)
datasetIO.save_datamatrix('C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_cluster_matrix_sarah.txt.gz', gene_cluster)
