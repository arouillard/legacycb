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
from sklearn.decomposition import PCA


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



# pca, euclidean distance
pca_model = PCA(n_components=2).fit(train.matrix)
pca2d_train = pca_model.transform(train.matrix)
pca2d_valid = pca_model.transform(valid.matrix)
pca2d_test = pca_model.transform(test.matrix)
with open('results/alternative_embeddings/pca2d.pickle', 'wb') as fw:
    pickle.dump((pca2d_train, pca2d_valid, pca2d_test), fw)

fg, ax = plt.subplots(1, 1, figsize=(6.5,6.5))
ax.set_position([0.15/6.5, 0.15/6.5, 6.2/6.5, 6.2/6.5])
ax.plot(pca2d_train[:,0], pca2d_train[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.plot(pca2d_valid[:,0], pca2d_valid[:,1], 'or', markersize=2, markeredgewidth=0, alpha=1.0, zorder=1)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/alternative_embeddings/pca2d.png', transparent=True, pad_inches=0, dpi=600)
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
    ax.plot(np.append(pca2d_train[hitt,0], pca2d_valid[hitv,0], 0), np.append(pca2d_train[hitt,1], pca2d_valid[hitv,1], 0), linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label=tissue)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, markerscale=4, fontsize=8, labelspacing=0.4)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
ax.set_frame_on(False)
fg.savefig('results/alternative_embeddings/pca2d_coloredbytissueenrichment.png', transparent=True, pad_inches=0, dpi=600)
plt.close()
