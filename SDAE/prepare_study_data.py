# -*- coding: utf-8 -*-
"""
@author: ar988996
"""


import numpy as np
import datasetIO
import dataclasses
import os
#from matplotlib import pyplot as plt
import sys


def main(study_name='your_study'):

    # load your data and create datamatrix object
    with open('data/original_data/{0}/ensembl_gene_ids.txt'.format(study_name), mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        ensembl_gene_ids = np.array([x.strip() for x in fr.read().split('\n')], dtype='object')
    
    with open('data/original_data/{0}/sample_ids.txt'.format(study_name), mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        sample_ids = np.array([x.strip() for x in fr.read().split('\n')], dtype='object')
    
    counts_matrix = np.loadtxt('data/original_data/{0}/expression_matrix.txt.gz'.format(study_name), dtype='float64', delimiter='\t', ndmin=2)
    total_counts_per_sample = counts_matrix.sum(0)
    
    gene_sample = dataclasses.datamatrix(rowname='ensembl_gene_id',
                                      rowlabels=ensembl_gene_ids,
                                      rowmeta={},
                                      columnname='sample_id',
                                      columnlabels=sample_ids,
                                      columnmeta={'total_counts':total_counts_per_sample},
                                      matrixname='rnaseq_gene_counts_from_{0}'.format(study_name),
                                      matrix=counts_matrix)
    del ensembl_gene_ids, sample_ids, counts_matrix, total_counts_per_sample
    
    
    # scale counts
    gene_sample.matrix = np.exp(np.log(gene_sample.matrix) - np.log(gene_sample.columnmeta['total_counts'].reshape(1,-1)) + (np.log(4) + 7*np.log(10)))
    gene_sample.matrixname = 'rnaseq_scaled_counts_from_{0}'.format(study_name)
    
    
    # shuffle the data
    gene_sample.reorder(np.random.permutation(gene_sample.shape[0]), 0)
    gene_sample.reorder(np.random.permutation(gene_sample.shape[1]), 1)
    print(gene_sample)
    
    
    # load the reference data
    gene_sample_ref = datasetIO.load_datamatrix('data/prepared_data/fat/train.pickle').totranspose()
    print(gene_sample_ref)
    
    
    # align genes
    tobediscarded = ~np.in1d(gene_sample.rowlabels, gene_sample_ref.rowmeta['ensembl_gene_id'])
    gene_sample.discard(tobediscarded, 0)
    missing_ensembl_ids = gene_sample_ref.rowmeta['ensembl_gene_id'][~np.in1d(gene_sample_ref.rowmeta['ensembl_gene_id'], gene_sample.rowlabels)]
    gene_sample = gene_sample.tolabels(rowlabels=gene_sample_ref.rowmeta['ensembl_gene_id'].copy(), columnlabels=[])
    gene_sample.rowlabels = gene_sample_ref.rowlabels.copy()
    gene_sample.rowname = gene_sample_ref.rowname
    for k, v in gene_sample_ref.rowmeta.items():
        gene_sample.rowmeta[k] = v.copy()
    gene_sample.rowmeta['is_missing'] = np.in1d(gene_sample.rowmeta['ensembl_gene_id'], missing_ensembl_ids)
    gene_sample.rowmeta['all_zero'] = (gene_sample.matrix == 0).all(1)
    print('missing data for {0!s} genes'.format(gene_sample.rowmeta['is_missing'].sum()))
    print('no counts for {0!s} genes'.format(gene_sample.rowmeta['all_zero'].sum()))
    print(gene_sample)
    
    
    # handle zeros
    nonzeromins = np.zeros(gene_sample.shape[1], dtype='float64')
    for j in range(gene_sample.shape[1]):
        nonzeromins[j] = gene_sample.matrix[gene_sample.matrix[:,j]>0,j].min()
        gene_sample.matrix[gene_sample.matrix[:,j]==0,j] = nonzeromins[j]/2.0
    
    
    # distributions
#    plt.figure(); plt.hist(gene_sample.matrix[:,:5], 50)
#    plt.figure(); plt.hist(((gene_sample.matrix[:5,:] - gene_sample.matrix[:5,:].mean(1, keepdims=True))/gene_sample.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)
    
    
    # log2
    gene_sample.matrix = np.log2(gene_sample.matrix)
    
    
    # distributions
#    plt.figure(); plt.hist(gene_sample.matrix[:,:5], 50)
#    plt.figure(); plt.hist(((gene_sample.matrix[:5,:] - gene_sample.matrix[:5,:].mean(1, keepdims=True))/gene_sample.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)
    
    
    # normalize samples
    median_shift_from_median = np.median(gene_sample.matrix - gene_sample.rowmeta['median_sample_ref'].reshape(-1,1), 0)
    gene_sample.matrix -=  median_shift_from_median.reshape(1,-1)
    
    
    # distributions
#    plt.figure(); plt.hist(gene_sample.matrix[:,:5], 50)
#    plt.figure(); plt.hist(((gene_sample.matrix[:5,:] - gene_sample.matrix[:5,:].mean(1, keepdims=True))/gene_sample.matrix[:5,:].std(1, ddof=1, keepdims=True)).T, 10)
    
    
    # standardize the data
    gene_sample.matrix = (gene_sample.matrix - gene_sample.rowmeta['row_mean_ref'].reshape(-1,1))/gene_sample.rowmeta['row_stdv_ref'].reshape(-1,1)
    
    
    # handle missing genes
    gene_sample.matrix[gene_sample.rowmeta['is_missing'],:] = 0
#    gene_sample.matrix[gene_sample.rowmeta['is_missing'],:] = gene_sample_ref.matrix[gene_sample.rowmeta['is_missing'],:].min(1, keepdims=True)/2.0
    
    
    # distributions
#    plt.figure(); plt.hist(gene_sample.matrix[:,:5], 50)
#    plt.figure(); plt.hist(gene_sample.matrix[:5,:].T, 10)
#    plt.figure(); plt.hist(gene_sample.matrix.reshape(-1), 1000)
    
    
    # transpose the data
    atb_gene = gene_sample.totranspose()
    
    
    # split the data
    test_fraction = 0.1
    tobepopped = np.random.permutation(gene_sample.shape[0]) < round(max([test_fraction*gene_sample.shape[0], 2.0]))
    gene_sample_test = gene_sample.pop(tobepopped, 0)
    valid_fraction = 0.1
    tobepopped = np.random.permutation(gene_sample.shape[0]) < round(max([valid_fraction*gene_sample.shape[0], 2.0]))
    gene_sample_valid = gene_sample.pop(tobepopped, 0)
    gene_sample_train = gene_sample
    del gene_sample, tobepopped
    
    
    # save the data
    if not os.path.exists('data/prepared_data'):
        os.mkdir('data/prepared_data')
    if not os.path.exists('data/prepared_data/{0}'.format(study_name)):
        os.mkdir('data/prepared_data/{0}'.format(study_name))
    if not os.path.exists('data/prepared_data/{0}/skinny'.format(study_name)):
        os.mkdir('data/prepared_data/{0}/skinny'.format(study_name))
    datasetIO.save_datamatrix('data/prepared_data/{0}/skinny/test.pickle'.format(study_name), gene_sample_test)
    datasetIO.save_datamatrix('data/prepared_data/{0}/skinny/valid.pickle'.format(study_name), gene_sample_valid)
    datasetIO.save_datamatrix('data/prepared_data/{0}/skinny/train.pickle'.format(study_name), gene_sample_train)
    del gene_sample_test, gene_sample_valid, gene_sample_train
    
    
    # split the data
    test_fraction = 0.1
    tobepopped = np.random.permutation(atb_gene.shape[0]) < round(max([test_fraction*atb_gene.shape[0], 2.0]))
    atb_gene_test = atb_gene.pop(tobepopped, 0)
    valid_fraction = 0.1
    tobepopped = np.random.permutation(atb_gene.shape[0]) < round(max([valid_fraction*atb_gene.shape[0], 2.0]))
    atb_gene_valid = atb_gene.pop(tobepopped, 0)
    atb_gene_train = atb_gene
    del atb_gene, tobepopped
    
    
    # save the data
    if not os.path.exists('data/prepared_data'):
        os.mkdir('data/prepared_data')
    if not os.path.exists('data/prepared_data/{0}'.format(study_name)):
        os.mkdir('data/prepared_data/{0}'.format(study_name))
    if not os.path.exists('data/prepared_data/{0}/fat'.format(study_name)):
        os.mkdir('data/prepared_data/{0}/fat'.format(study_name))
    datasetIO.save_datamatrix('data/prepared_data/{0}/fat/test.pickle'.format(study_name), atb_gene_test)
    datasetIO.save_datamatrix('data/prepared_data/{0}/fat/valid.pickle'.format(study_name), atb_gene_valid)
    datasetIO.save_datamatrix('data/prepared_data/{0}/fat/train.pickle'.format(study_name), atb_gene_train)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError('too many inputs')

