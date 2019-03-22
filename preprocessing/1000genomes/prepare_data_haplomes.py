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
    genome_ids_ = []
    pops_ = []
    super_pops_ = []
    genders_ = []
    for line in fr:
        genome_id, pop, super_pop, gender = [x.strip() for x in line.split('\t')]
        genome_ids_.append(genome_id)
        pops_.append(pop)
        super_pops_.append(super_pop)
        genders_.append(gender)
gid_idx = {g:i for i,g in enumerate(genome_ids_)}

# get gene positions
print('getting gene positions...', flush=True)
gene_meta_fields = ['gene_stable_id', 'chromosome_scaffold_name', 'gene_start_bp', 'gene_end_bp', 'gene_name', 'gene_type', 'entrezgene_id']
gene_meta_dtypes = ['object', 'object', 'float64', 'float64', 'object', 'object', 'object']
gene_meta_data = {f:np.zeros(0, dtype=d) for f,d in zip(gene_meta_fields, gene_meta_dtypes)}
with open('../../original_data/1000genomes/MHC_Region_Gene_Positions_GeneHasEntrezID.txt', 'rt') as fr:
    headerline = fr.readline()
    for line in fr:
        values = [x.strip() for x in line.split('\t')]
        if values[-2] == 'protein_coding':
            for f,v in zip(gene_meta_fields, values):
                gene_meta_data[f] = np.insert(gene_meta_data[f], gene_meta_data[f].size, v)

# count overlapping genes
print('counting overlapping genes...', flush=True)
is_overlap = np.logical_and(gene_meta_data['gene_start_bp'].reshape(-1,1) >= gene_meta_data['gene_start_bp'].reshape(1,-1), gene_meta_data['gene_start_bp'].reshape(-1,1) < gene_meta_data['gene_end_bp'].reshape(1,-1))
print('overlapping genes: {0!s}'.format(is_overlap.sum()), flush=True)

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
        genome_ids = [x.strip() for x in headerline.split('\t')][9:]
        haplome_ids = '\t'.join(['{0}_0\t{0}_1'.format(x) for x in genome_ids]).split('\t')
        genome_ids = '\t'.join(['{0}\t{0}'.format(x) for x in genome_ids]).split('\t')
        pops = []
        super_pops = []
        genders = []
        for gid in genome_ids:
            i = gid_idx[gid]
            pops.append(pops_[i])
            super_pops.append(super_pops_[i])
            genders.append(genders_[i])
        rsids = []
        chroms = []
        poss = []
        refs = []
        alts = []
        ensembl_gene_ids = []
        gene_names = []
        gene_types = []
        entrez_ids = []
        genotype_matrix = []
        for line in fr:
            chrom, pos, rsid, ref, alt, qual, filt, info, fmt, genotypes = [x.strip() for x in line.split('\t', maxsplit=9)]
            if qual == '100' and filt == 'PASS' and 'VT=SNP' in info and 'MULTI_ALLELIC' not in info:
                genotype_matrix.append([float(x.strip()) for x in genotypes.replace('|','\t').split('\t')])
                rsids.append(rsid)
                chroms.append(chrom)
                poss.append(pos)
                refs.append(ref)
                alts.append(alt)
                gene_hit = np.logical_and(float(pos) >= gene_meta_data['gene_start_bp'], float(pos) < gene_meta_data['gene_end_bp'])
                if gene_hit.any():
                    hidxs = gene_hit.nonzero()[0]
                    ensembl_gene_ids.append('|'.join(gene_meta_data['gene_stable_id'][hidxs]))
                    gene_names.append('|'.join(gene_meta_data['gene_name'][hidxs]))
                    gene_types.append('|'.join(gene_meta_data['gene_type'][hidxs]))
                    entrez_ids.append('|'.join(gene_meta_data['entrezgene_id'][hidxs]))
                    if gene_hit.sum() > 1:
                        print('warning: {0!s} genes overlap with {1}'.format(gene_hit.sum(), rsid), flush=True)
                        print(ensembl_gene_ids[-1], gene_names[-1], gene_types[-1], entrez_ids[-1], flush=True)
                else:
                    ensembl_gene_ids.append('NA')
                    gene_names.append('NA')
                    gene_types.append('NA')
                    entrez_ids.append('NA')
    
    print('creating datamatrix object...', flush=True)
    dataset[partition] = dc.datamatrix(rowname='rsid',
                                       rowlabels=np.array(rsids, dtype='object'),
                                       rowmeta={'chromosome':np.array(chroms, dtype='object'), 'position':np.array(poss, dtype='object'), 'ref_allele':np.array(refs, dtype='object'), 'alt_allele':np.array(alts, dtype='object'), 'ensembl_gene_id':np.array(ensembl_gene_ids, dtype='object'), 'gene_name':np.array(gene_names, dtype='object'), 'gene_type':np.array(gene_types, dtype='object'), 'entrez_id':np.array(entrez_ids, dtype='object')},
                                       columnname='haplome_id',
                                       columnlabels=np.array(haplome_ids, dtype='object'),
                                       columnmeta={'genome_id':np.array(genome_ids, dtype='object'), 'population':np.array(pops, dtype='object'), 'super_population':np.array(super_pops, dtype='object'), 'gender':np.array(genders, dtype='object')},
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
    
    print('merging datamatrix objects...', flush=True)
    if 'all' not in dataset:
        dataset['all'] = copy.deepcopy(dataset[partition])
    else:
        dataset['all'].append(dataset[partition], 0)
    del dataset[partition]

# rename datamatrix object
print('done reading VCFs...', flush=True)
snp_haplome = dataset['all']
del dataset
print(snp_haplome, flush=True)

# discard constant SNPs
print('discarding constant SNPs...', flush=True)
tobediscarded = (snp_haplome.matrix == snp_haplome.matrix[:,0].reshape(-1,1)).all(1)
snp_haplome.discard(tobediscarded, 0)
print(snp_haplome, flush=True)

# discard constant haplomes (shouldn't be any)
print('discarding constant haplomes...', flush=True)
tobediscarded = (snp_haplome.matrix == snp_haplome.matrix[0,:].reshape(1,-1)).all(0)
snp_haplome.discard(tobediscarded, 1)
print(snp_haplome, flush=True)

# add SNP mean (frequency) and stdv (sqrt(p(1-p))) to metadata
print('adding SNP mean (frequency) and stdv to metadata...', flush=True)
snp_haplome.rowmeta['mean'] = snp_haplome.matrix.mean(1) 
snp_haplome.rowmeta['stdv'] = snp_haplome.matrix.std(1)

# shuffle the data
print('shuffling data...', flush=True)
snp_haplome.reorder(np.random.permutation(snp_haplome.shape[0]), 0)
snp_haplome.reorder(np.random.permutation(snp_haplome.shape[1]), 1)
print(snp_haplome, flush=True)

# save the data
print('saving prepared data...', flush=True)
snp_haplome.matrixname += '_prepared'
datasetIO.save_datamatrix('../../original_data/1000genomes/snp_haplome_1000genomes-phased-MHC_prepared.pickle', snp_haplome)
datasetIO.save_datamatrix('../../original_data/1000genomes/snp_haplome_1000genomes-phased-MHC_prepared.txt.gz', snp_haplome)
savefolder = '../../input_data/1000genomes_haplomes'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, snp_haplome)
shutil.copyfile('../../original_data/1000genomes/snp_haplome_1000genomes-phased-MHC_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/1000genomes/snp_haplome_1000genomes-phased-MHC_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)


# visualization
pca_model = PCA(n_components=2).fit(snp_haplome.matrix)
pca_matrix = pca_model.transform(snp_haplome.matrix)
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.plot(pca_matrix[:,0], pca_matrix[:,1], 'ok', markersize=1, markeredgewidth=0, alpha=0.5, zorder=0)
ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/snp_haplome_pca.png', transparent=True, pad_inches=0, dpi=300)
plt.close()
for metalabel in ['mean', 'stdv', 'position']:
    z = snp_haplome.rowmeta[metalabel].astype('float64')
    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
    ax.scatter(pca_matrix[:,0], pca_matrix[:,1],  s=1, c=z, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), alpha=0.5, vmin=z.min(), vmax=z.max())
    ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
    ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
    ax.set_frame_on(False)
    fg.savefig('../../original_data/1000genomes/snp_haplome_pca_colored_by_{0}.png'.format(metalabel), transparent=True, pad_inches=0, dpi=300)
    plt.close()
hla_hit = np.array(['HLA-' in x for x in snp_haplome.rowmeta['gene_name']], dtype='bool')
hla_names = snp_haplome.rowmeta['gene_name'].copy()
hla_names[~hla_hit] = 'NA'
categories = np.unique(hla_names)
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
for category, color in zip(categories, colors):
    if category == 'NA':
        color = 'k'
        alpha = 0.1
        zorder = 0
    else:
        alpha = 0.5
        zorder = 1
    hit = hla_names == category
    ax.plot(pca_matrix[hit,0], pca_matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=1, markeredgewidth=0, alpha=alpha, zorder=zorder, label=category)
ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/snp_haplome_pca_colored_by_hlagene.png', transparent=True, pad_inches=0, dpi=300)
plt.close()

pca_model = PCA(n_components=2).fit(snp_haplome.matrix.T)
pca_matrix = pca_model.transform(snp_haplome.matrix.T)
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.plot(pca_matrix[:,0], pca_matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/haplome_snp_pca.png', transparent=True, pad_inches=0, dpi=300)
plt.close()
for metalabel in ['population', 'super_population', 'gender']:
    categories = np.unique(snp_haplome.columnmeta[metalabel])
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
    for category, color in zip(categories, colors):
        hit = snp_haplome.columnmeta[metalabel] == category
        ax.plot(pca_matrix[hit,0], pca_matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=0.5, zorder=0, label=category)
    ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
    ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
    ax.set_frame_on(False)
    fg.savefig('../../original_data/1000genomes/haplome_snp_pca_colored_by_{0}.png'.format(metalabel), transparent=True, pad_inches=0, dpi=300)
    plt.close()
