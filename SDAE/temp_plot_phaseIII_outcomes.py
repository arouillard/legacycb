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

# load class examples
print('loading class examples...', flush=True)
class_examples_folder = 'C:/Users/ar988996/Documents/omic-features-successful-targets/targets/pharmaprojects'
class_examples = {'positive':datasetIO.load_examples('{0}/positive.txt'.format(class_examples_folder)),
                  'negative':datasetIO.load_examples('{0}/negative.txt'.format(class_examples_folder)),
                  'unknown':datasetIO.load_examples('{0}/unknown.txt'.format(class_examples_folder))}

# assign class labels to genes
print('assigning class labels to genes...', flush=True)
gene_proj.rowmeta['class'] = np.full(gene_proj.shape[0], 'unknown', dtype='object')
gene_proj.rowmeta['class'][np.in1d(gene_proj.rowlabels, list(class_examples['positive']))] = 'positive'
gene_proj.rowmeta['class'][np.in1d(gene_proj.rowlabels, list(class_examples['negative']))] = 'negative'

# plot
#yub = 0.5
#ylb = -0.35
#xub = 0.45
#xlb = -0.4
#fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
#ax.set_position([0.25/6.5, 0.25/6.5, 6/6.5, 6/6.5]) # left, bottom, width, height
#ax.plot(gene_proj.matrix[gene_proj.rowmeta['class']=='positive',0], gene_proj.matrix[gene_proj.rowmeta['class']=='positive',1], 'sr', markersize=3, markeredgewidth=0, alpha=1, zorder=1, label='success')
#ax.plot(gene_proj.matrix[gene_proj.rowmeta['class']=='negative',0], gene_proj.matrix[gene_proj.rowmeta['class']=='negative',1], 'xb', markersize=3, markeredgewidth=1, alpha=1, zorder=2, label='failure')
#ax.plot(gene_proj.matrix[gene_proj.rowmeta['class']=='unknown',0], gene_proj.matrix[gene_proj.rowmeta['class']=='unknown',1], 'ok', markersize=3, markeredgewidth=0, alpha=0.01, zorder=0, label='unknown')
#ax.set_ylim(ylb, yub)
#ax.set_xlim(xlb, xub)
#ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
#ax.set_frame_on(False)
#ax.legend(loc='lower left', ncol=1, fontsize=8, frameon=False, borderpad=5, labelspacing=0.1, handletextpad=0.1, borderaxespad=0)
#fg.savefig('temp_plot_phaseIII_outcomes.png', transparent=True, pad_inches=0, dpi=600)

'''
yub = 0.45
ylb = -0.35
xub = 0.55
xlb = -0.4
fg, ax = plt.subplots(1, 1, figsize=(3.25,2))
ax.set_position([0.5/3.25, 0.35/2, 2.5/3.25, 1.55/2]) # left, bottom, width, height
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['class']=='positive',0], gene_proj.matrix[gene_proj.rowmeta['class']=='positive',1], s=5, c='r', marker='+', alpha=1, edgecolors='r', linewidths=0.5, zorder=1, label='success')
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['class']=='negative',0], gene_proj.matrix[gene_proj.rowmeta['class']=='negative',1], s=4, c='b', marker='x', alpha=1, edgecolors='b', linewidths=0.5, zorder=2, label='failure')
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['class']=='unknown',0], gene_proj.matrix[gene_proj.rowmeta['class']=='unknown',1], s=1, c='k', marker='o', alpha=0.1, edgecolors='k', linewidths=0, zorder=0, label='unknown')
ax.set_xlabel('SDAE-1', fontsize=8, labelpad=2)
ax.set_ylabel('SDAE-2', fontsize=8, labelpad=4)
ax.tick_params(axis='both', which='major', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off', labelsize=8)
ax.legend(loc='center right', ncol=1, fontsize=8, frameon=False, borderpad=0.1, labelspacing=0, handletextpad=0, borderaxespad=0)
ax.set_ylim(ylb, yub)
ax.set_xlim(xlb, xub)
fg.savefig('temp_plot_phaseIII_outcomes_formatted.png', dpi=600, transparent=True, pad_inches=0)
'''



yub = 0.45
ylb = -0.35
xub = 0.55
xlb = -0.4
fg, ax = plt.subplots(1, 1, figsize=(9.75,6))
ax.set_position([1.5/9.75, 1.05/6, 7.5/9.75, 4.7/6]) # left, bottom, width, height
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['class']=='positive',0], gene_proj.matrix[gene_proj.rowmeta['class']=='positive',1], s=20, c='r', marker='+', alpha=1, edgecolors='r', linewidths=1.5, zorder=1, label='success')
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['class']=='negative',0], gene_proj.matrix[gene_proj.rowmeta['class']=='negative',1], s=16, c='b', marker='x', alpha=1, edgecolors='b', linewidths=1.5, zorder=2, label='failure')
ax.scatter(gene_proj.matrix[gene_proj.rowmeta['class']=='unknown',0], gene_proj.matrix[gene_proj.rowmeta['class']=='unknown',1], s=1, c='k', marker='o', alpha=0.1, edgecolors='k', linewidths=0, zorder=0, label='unknown')
ax.set_xlabel('SDAE-1', fontsize=16, labelpad=2)
ax.set_ylabel('SDAE-2', fontsize=16, labelpad=4)
ax.tick_params(axis='both', which='major', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off', labelsize=16)
ax.legend(loc='upper left', ncol=1, fontsize=16, frameon=False, borderpad=0.1, labelspacing=0, handletextpad=0, borderaxespad=0)
ax.set_ylim(ylb, yub)
ax.set_xlim(xlb, xub)
fg.savefig('temp_plot_phaseIII_outcomes_formatted_big.png', dpi=600, transparent=True, pad_inches=0)

hit = np.logical_and(np.logical_and(gene_proj.matrix[:,0] > 0.05, gene_proj.matrix[:,1] > -0.01), gene_proj.rowmeta['class'] != 'unknown')
selected = copy.deepcopy(gene_proj)
selected.discard(~hit, 0)
for g,i,c,t in zip(selected.rowlabels, selected.rowmeta['GeneID'], selected.rowmeta['class'], selected.rowmeta['Tissue']):
    print('{0},{1},{2},{3}'.format(g, i, c, t))
print('\n\n\n')
hit = np.logical_and(gene_proj.matrix[:,0] < -0.1, gene_proj.rowmeta['class'] != 'unknown')
selected = copy.deepcopy(gene_proj)
selected.discard(~hit, 0)
for g,i,c,t in zip(selected.rowlabels, selected.rowmeta['GeneID'], selected.rowmeta['class'], selected.rowmeta['Tissue']):
    print('{0},{1},{2},{3}'.format(g, i, c, t))

datasetIO.save_datamatrix('temp_gene_proj_phaseIII_outcomes.txt.gz', gene_proj)
