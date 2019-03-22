# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:36:25 2016

@author: ar988996
"""

import numpy as np
import gzip

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

# write truncated file
nmax = 10
with open('average_pearson_similarity_top{!s}.txt'.format(nmax), 'wt') as fw:
    for i, genesym in enumerate(gene_symbol):
        geneid = gene_id[i]
        ensemblid = ensembl_id[i]
        si = np.argsort(matrix[i,:])[::-1][1:nmax + 1]
        fw.write('\t'.join([genesym, geneid, ensemblid] + ['{0},{1:1.3g}'.format(gene_symbol[j], matrix[i,j]) for j in si]) + '\n')
