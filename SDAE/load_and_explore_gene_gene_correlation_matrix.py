# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:36:25 2016

@author: ar988996
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
from machinelearning import datasetIO

distance_sigma = 0.01 # 0.05 # 0.0011 is mean of min gene-gene distance
gene_gene = datasetIO.load_datamatrix('../gene_gene_matrix_euclidean_distance_from_projection.pickle')
gene_gene.matrix = np.exp(-1*gene_gene.matrix**2/2/distance_sigma**2)

gene_gene = gene_gene.toCLR()

# write truncated file
nmax = 100
with open('autoencoder_similarity_clr_normalized_top{!s}.txt'.format(nmax), 'wt') as fw:
    for i, genesym in enumerate(gene_gene.rowlabels):
        geneid = gene_gene.rowmeta['GeneID'][i]
        ensemblid = gene_gene.rowmeta['Ensemble Acc'][i]
        si = np.argsort(gene_gene.matrix[i,:])[::-1][1:nmax + 1]
        fw.write('\t'.join([genesym, geneid, ensemblid] + ['{0},{1:1.6g}'.format(gene_gene.columnlabels[j], gene_gene.matrix[i,j]) for j in si]) + '\n')
























'''
# load tab-delimited matrix
# first three header lines begin with #, these are automatically recognized as comment lines and skipped by the np.loadtxt reader
# takes 5-10 minutes to load
matrix = np.loadtxt('average_pearson_similarity.txt.gz', dtype='float64', comments='#', delimiter='\t')

# parse row/column labels from header lines
# the matrix is symmetric, so the row and column labels are the same
with gzip.open('average_pearson_similarity.txt.gz', 'rt') as fr:
    gene_symbol = np.array(fr.readline().strip().split('\t')[1:], dtype='object')
    gene_id = np.array(fr.readline().strip().split('\t')[1:], dtype='object')
    ensembl_id = np.array(fr.readline().strip().split('\t')[1:], dtype='object')

gene_gene = dc.datamatrix(rowname='gene',
                          rowlabels=gene_symbol.copy(),
                          rowmeta={'gene_id':gene_id.copy(), 'ensembl_id':ensembl_id.copy()},
                          columnname='gene',
                          columnlabels=gene_symbol.copy(),
                          columnmeta={'gene_id':gene_id.copy(), 'ensembl_id':ensembl_id.copy()},
                          matrixname='average_pearson_simlarity',
                          matrix=matrix)
del matrix, gene_symbol, gene_id, ensembl_id

gene_gene_clr = gene_gene.toCLR()

# write truncated file
#nmax = 100
#with open('average_pearson_similarity_clr_normalized_top{!s}.txt'.format(nmax), 'wt') as fw:
#    for i, genesym in enumerate(gene_gene_clr.rowlabels):
#        geneid = gene_gene_clr.rowmeta['gene_id'][i]
#        ensemblid = gene_gene_clr.rowmeta['ensembl_id'][i]
#        si = np.argsort(gene_gene_clr.matrix[i,:])[::-1][1:nmax + 1]
#        fw.write('\t'.join([genesym, geneid, ensemblid] + ['{0},{1:1.6g}'.format(gene_gene_clr.columnlabels[j], gene_gene_clr.matrix[i,j]) for j in si]) + '\n')


#gene_gene.cluster('symmetric')
#for i in range(gene_gene.shape[0]):
#    gene_gene.matrix[i,i] = 0
##set ub and lb
#gene_gene.bigheatmap(rowmetalabels=[], columnmetalabels=[], savefilename='average_pearson_similarity_heatmap.png')

#gene_gene_known = ds.loaddatamatrix(datasetpath='C:/Users/ar988996/Documents/Harmonizome/datasets/biogrid/gene_attribute_matrix.txt.gz',
#                                          rowname='gene',
#                                          columnname='gene',
#                                          matrixname='protein-protein interactions',
#                                          skiprows=3,
#                                          skipcolumns=3,
#                                          delimiter='\t',
#                                          dtype='int8',
#                                          getmetadata=True)
#gene_gene_known = ds.loaddatamatrix(datasetpath='C:/Users/ar988996/Documents/Harmonizome/datasets/intact/gene_attribute_matrix.txt.gz',
#                                          rowname='gene',
#                                          columnname='gene',
#                                          matrixname='protein-protein interactions',
#                                          skiprows=3,
#                                          skipcolumns=3,
#                                          delimiter='\t',
#                                          dtype='int8',
#                                          getmetadata=True)
with open('C:/Users/ar988996/Documents/Metabase/gene_gene_metabase.pickle', 'rb') as fr:
    gene_gene_known = pickle.load(fr)

commongenes = np.array(list(set(gene_gene.rowlabels).intersection(gene_gene_known.rowlabels).intersection(gene_gene.columnlabels).intersection(gene_gene_known.columnlabels)), dtype='object')
gg = gene_gene.tolabels(commongenes, commongenes)
ggc = gene_gene_clr.tolabels(commongenes, commongenes)
gene_gene_known = gene_gene_known.tolabels(commongenes, commongenes)
ltmask = np.tril(np.ones(gg.shape, dtype='bool'), -1)
gg = gg.matrix[ltmask]
ggc = ggc.matrix[ltmask]
ggk = gene_gene_known.matrix[ltmask] == 1
gg_performance = cp.summary(Y=ggk, P=gg, scores_are_probabilities=False, plot_curves=False, return_curves=True, downsample_curves=False, get_positive_cutoff=True)
ggc_performance = cp.summary(Y=ggk, P=ggc, scores_are_probabilities=False, plot_curves=False, return_curves=True, downsample_curves=False, get_positive_cutoff=True)

ggx_performance = cp.summary(Y=ggk, P=np.abs(gg), scores_are_probabilities=False, plot_curves=True, return_curves=False, downsample_curves=True, get_positive_cutoff=False)
'''