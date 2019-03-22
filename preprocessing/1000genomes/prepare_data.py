# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import dataclasses as dc
import datasetIO
import os
import shutil
import copy
import gzip
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# read the sample metadata
print('reading sample metadata...', flush=True)
with open('../../original_data/1000genomes/integrated_call_samples_v3.20130502.ALL.panel', 'rt') as fr:
    headerline = fr.readline()
    sample_ids_ = []
    pops_ = []
    super_pops_ = []
    genders_ = []
    for line in fr:
        sample, pop, super_pop, gender = [x.strip() for x in line.split('\t')]
        sample_ids_.append(sample)
        pops_.append(pop)
        super_pops_.append(super_pop)
        genders_.append(gender)
sid_idx = {g:i for i,g in enumerate(sample_ids_)}

# read phased genotypes from VCFs
print('reading phased genotypes from VCFs...', flush=True)
partitions = ['HLA_Part1.vcf.gz', 'HLA_Part2.vcf.gz', 'HLA_Part3.vcf.gz', 'HLA_Part4.vcf.gz', 'HLA_Part5.vcf.gz']
dataset = {}
for partition in partitions:
    partition_path = '../../original_data/1000genomes/{0}'.format(partition)
    print('working on partition_path: {0}'.format(partition_path), flush=True)
    with gzip.open(partition_path, 'rt') as fr:
        headerline = fr.readline()
        while headerline[:2] == '##':
            headerline = fr.readline()
        sample_ids = [x.strip() for x in headerline.split('\t')][9:]
        genome_ids = '\t'.join(['{0}_0\t{0}_1'.format(x) for x in sample_ids]).split('\t')
        sample_ids = '\t'.join(['{0}\t{0}'.format(x) for x in sample_ids]).split('\t')
        pops = []
        super_pops = []
        genders = []
        for s in sample_ids:
            i = sid_idx[s]
            pops.append(pops_[i])
            super_pops.append(super_pops_[i])
            genders.append(genders_[i])
        rsids = []
        chroms = []
        poss = []
        refs = []
        alts = []
        genotype_matrix = []
        for line in fr:
            chrom, pos, rsid, ref, alt, qual, filt, info, fmt, genotypes = [x.strip() for x in line.split('\t', maxsplit=9)]
            if qual == '100' and filt == 'PASS' and 'VT=SNP' in info:
                genotype_matrix.append([float(x.strip()) for x in genotypes.replace('|','\t').split('\t')])
                rsids.append(rsid)
                chroms.append(chrom)
                poss.append(pos)
                refs.append(ref)
                alts.append(alt)
    dataset[partition] = dc.datamatrix(rowname='rsid',
                                       rowlabels=np.array(rsids, dtype='object'),
                                       rowmeta={'chromosome':np.array(chroms, dtype='object'), 'position':np.array(poss, dtype='object'), 'ref_allele':np.array(refs, dtype='object'), 'alt_allele':np.array(alts, dtype='object')},
                                       columnname='genome_id',
                                       columnlabels=np.array(genome_ids, dtype='object'),
                                       columnmeta={'sample_id':np.array(sample_ids, dtype='object'), 'population':np.array(pops, dtype='object'), 'super_population':np.array(super_pops, dtype='object'), 'gender':np.array(genders, dtype='object')},
                                       matrixname='MHC_phased_genotypes_from_1000_genomes',
                                       matrix=np.array(genotype_matrix, dtype='float32'))
    print(dataset[partition], flush=True)
    for i in range(5):
        printdict = {dataset[partition].rowname:dataset[partition].rowlabels[i]}
        for k, v in dataset[partition].rowmeta.items():
            printdict[k] = v[i]
        print(printdict, flush=True)
    for i in range(5):
        printdict = {dataset[partition].columnname:dataset[partition].columnlabels[i]}
        for k, v in dataset[partition].columnmeta.items():
            printdict[k] = v[i]
        print(printdict, flush=True)
    if 'all' not in dataset:
        dataset['all'] = copy.deepcopy(dataset[partition])
    else:
        dataset['all'].append(dataset[partition], 0)
    del dataset[partition]

# rename datamatrix object
print('done reading VCFs...', flush=True)
snp_genome = dataset['all']
del dataset
print(snp_genome, flush=True)

# discard constant SNPs
print('discarding constant SNPs...', flush=True)
tobediscarded = (snp_genome.matrix == snp_genome.matrix[:,0].reshape(-1,1)).all(1)
snp_genome.discard(tobediscarded, 0)
print(snp_genome, flush=True)

# discard constant genomes (shouldn't be any)
print('discarding constant genomes...', flush=True)
tobediscarded = (snp_genome.matrix == snp_genome.matrix[0,:].reshape(1,-1)).all(0)
snp_genome.discard(tobediscarded, 1)
print(snp_genome, flush=True)

# shuffle the data
print('shuffling data...', flush=True)
snp_genome.reorder(np.random.permutation(snp_genome.shape[0]), 0)
snp_genome.reorder(np.random.permutation(snp_genome.shape[1]), 1)
print(snp_genome, flush=True)

# save the data
print('saving prepared data...', flush=True)
snp_genome.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/1000genomes/snp_genome_1000genomes-phased-MHC_prepared.pickle', snp_genome)
datasetIO.save_datamatrix('../../original_data/1000genomes/snp_genome_1000genomes-phased-MHC_prepared.txt.gz', snp_genome)
savefolder = '../../input_data/1000genomes'
os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, snp_genome)
shutil.copyfile('../../original_data/1000genomes/snp_genome_1000genomes-phased-MHC_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)



# visualization
pca_model = PCA(n_components=2).fit(snp_genome.matrix)
pca_matrix = pca_model.transform(snp_genome.matrix)
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.plot(pca_matrix[:,0], pca_matrix[:,1], 'ok', markersize=1, markeredgewidth=0, alpha=0.5, zorder=0)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off')
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/pca.png', transparent=True, pad_inches=0, dpi=300)
pos = snp_genome.rowmeta['position'].astype('float64')
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.scatter(pca_matrix[:,0], pca_matrix[:,1],  s=1, c=pos, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), vmin=pos.min(), vmax=pos.max())
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off')
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/pca_colored_by_position.png', transparent=True, pad_inches=0, dpi=300)

pca_model = PCA(n_components=2).fit(snp_genome.matrix.T)
pca_matrix = pca_model.transform(snp_genome.matrix.T)
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.plot(pca_matrix[:,0], pca_matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off')
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/pca_transpose.png', transparent=True, pad_inches=0, dpi=300)
for category_name in ['population', 'super_population', 'gender']:
    categories = np.unique(snp_genome.columnmeta[category_name])
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
    for category, color in zip(categories, colors):
        hit = snp_genome.columnmeta[category_name] == category
        ax.plot(pca_matrix[hit,0], pca_matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=0.5, zorder=0, label=category)
    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
    ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
    ax.set_frame_on(False)
    fg.savefig('../../original_data/1000genomes/pca_transpose_colored_by_{0}.png'.format(category_name), transparent=True, pad_inches=0, dpi=300)
