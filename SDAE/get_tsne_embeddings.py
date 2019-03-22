# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
#custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
#               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
#               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
#               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


def get_distance_matrix(X, metric):
    if metric == 'euclidean':
        return euclidean_distances(X)
    elif metric == 'angular_cosine':
        D = cosine_similarity(X)
        D[D<-1] = -1
        D[D>1] = 1
        if D.min() < 0:
            D = np.arccos(D)/np.pi # divide by pi if similarity scores can be negative, otherwise divide by pi/2
        else:
            D = np.arccos(D)/(np.pi/2)
        return D
    else:
        raise ValueError('invalid distance metric')


# load the data
with open('data/prepared_data/train.pickle', 'rb') as fr:
    train = pickle.load(fr)
with open('data/prepared_data/valid.pickle', 'rb') as fr:
    valid = pickle.load(fr)
with open('data/prepared_data/test.pickle', 'rb') as fr:
    test = pickle.load(fr)
train_examples = train.shape[0]
valid_examples = valid.shape[0]
test_examples = test.shape[0]


# load annotations
with open('data/original_data/geneid_tissue_mark.pickle', 'rb') as fr:
    geneid_tissue = pickle.load(fr)
train.rowmeta['Tissue'] = np.full(train.shape[0], 'none', dtype='object')
for i, geneid in enumerate(train.rowmeta['GeneID']):
    if geneid in geneid_tissue:
        train.rowmeta['Tissue'][i] = geneid_tissue[geneid]
valid.rowmeta['Tissue'] = np.full(valid.shape[0], 'none', dtype='object')
for i, geneid in enumerate(valid.rowmeta['GeneID']):
    if geneid in geneid_tissue:
        valid.rowmeta['Tissue'][i] = geneid_tissue[geneid]
tissues = ['Brain', 'Pituitary', 'Spleen', 'Blood', 'Bone Marrow', 'Testis', 'Liver', 'Kidney', 'Salivary Gland', 'Heart', 'Muscle', 'Pancreas', 'Stomach', 'Small Intestine', 'Colon', 'Fallopian Tube', 'Skin', 'Bladder', 'Cervix Uteri', 'Prostate', 'Esophagus', 'Vagina', 'Adrenal Gland', 'Uterus', 'Ovary', 'Breast', 'Nerve', 'Adipose Tissue', 'Blood Vessel', 'Lung', 'Thyroid', 'none']
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/len(tissues))) for i in range(len(tissues))]


# create output directories
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('results/alternative_embeddings'):
    os.mkdir('results/alternative_embeddings')
if not os.path.exists('results/alternative_embeddings'):
    os.mkdir('results/alternative_embeddings')


# specify hyperparameters
perplexities = [10.0, 30.0, 50.0, 70.0]
early_exaggerations = [1.0, 4.0, 10.0]
learning_rates = [100.0, 300.0, 1000.0, 3000.0]
distance_metrics = ['euclidean', 'angular_cosine']
num_reps = 3

# calculate embeddings
for distance_metric in distance_metrics:
    for perplexity in perplexities:
        for early_exaggeration in early_exaggerations:
            for learning_rate in learning_rates:
                for rep in range(num_reps):
                    if os.path.exists('results/alternative_embeddings/tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}_in_progress.txt'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep)):
                        continue
                    print('working on tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep))
                    with open('results/alternative_embeddings/tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}_in_progress.txt'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep), 'wt') as fw:
                        fw.write('working on tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep))
                    
                    D = get_distance_matrix(np.concatenate((train.matrix, valid.matrix, test.matrix), 0), distance_metric)
                    tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=1000, metric='precomputed', verbose=1).fit(D)
                    tsne2d_train = tsne.embedding_[:train_examples,:]
                    tsne2d_valid = tsne.embedding_[train_examples:train_examples+valid_examples,:]
                    tsne2d_test = tsne.embedding_[train_examples+valid_examples:,:]
                    with open('results/alternative_embeddings/tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}.pickle'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep), 'wb') as fw:
                        pickle.dump((tsne2d_train, tsne2d_valid, tsne2d_test, tsne.kl_divergence_), fw)
                    
                    fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
                    ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
                    ax.plot(tsne2d_train[:,0], tsne2d_train[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
                    ax.plot(tsne2d_valid[:,0], tsne2d_valid[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
                    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
                    ax.set_frame_on(False)
                    fg.savefig('results/alternative_embeddings/tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}.png'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep), transparent=True, pad_inches=0, dpi=600)
                    plt.close()
                    
                    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                    for tissue, color in zip(tissues, colors):
                        if tissue == 'none':
                            zorder = 0
                            alpha = 0.05
                            color = 'k'
                        else:
                            zorder = 1
                            alpha = 0.5
                        hitt = train.rowmeta['Tissue'] == tissue
                        hitv = valid.rowmeta['Tissue'] == tissue
                        ax.plot(np.append(tsne2d_train[hitt,0], tsne2d_valid[hitv,0], 0), np.append(tsne2d_train[hitt,1], tsne2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
                    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
                    ax.set_frame_on(False)
                    fg.savefig('results/alternative_embeddings/tsne2d_{0}_p{1!s}_ee{2!s}_lr{3!s}_r{4!s}_coloredbytissueenrichment_ccle.png'.format(distance_metric, perplexity, early_exaggeration, learning_rate, rep), transparent=True, pad_inches=0, dpi=600)
                    plt.close()
                    
                    del D, tsne, tsne2d_train, tsne2d_valid, tsne2d_test


