# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
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
from scipy import stats

# load testdata
print('loading test datamatrix...', flush=True)
ds = datasetIO.load_datamatrix('../../partitioned_data/GTEXv6plusTL/fat/test.pickle')
print(ds, flush=True)

meta_labels = ['general_tissue', 'specific_tissue']
num_reps = 100
NUM_SAMPLES = np.array([3, 10, 30, 100, 300, 1000, 3000, 10000], dtype='int64')
for meta_label in meta_labels:
    categories = np.unique(ds.rowmeta[meta_label])
    for i, cati in enumerate(categories[:-1]):
        for catj in categories[i+1:]:
            subds = copy.deepcopy(ds)
            tobediscarded = ~np.in1d(subds.rowmeta[meta_label], [cati, catj])
            subds.discard(tobediscarded, 0)
            print(subds, flush=True)
            idxsi = (subds.rowmeta[meta_label] == cati).nonzero()[0]
            idxsj = (subds.rowmeta[meta_label] == catj).nonzero()[0]
            max_samples = np.min([idxsi.size, idxsj.size])
            num_samples = NUM_SAMPLES[NUM_SAMPLES < max_samples]
            num_samples = np.insert(num_samples, num_samples.size, max_samples)
            tvalues_mean = np.zeros((num_samples.size, subds.shape[1]), dtype='float64')
            tvalues_stdv = np.zeros((num_samples.size, subds.shape[1]), dtype='float64')
            pvalues_mean = np.zeros((num_samples.size, subds.shape[1]), dtype='float64')
            pvalues_stdv = np.zeros((num_samples.size, subds.shape[1]), dtype='float64')
            pc1values_mean = np.zeros((num_samples.size, subds.shape[1]), dtype='float64')
            pc1values_stdv = np.zeros((num_samples.size, subds.shape[1]), dtype='float64')
            msevalues_mean = np.zeros(num_samples.size, dtype='float64')
            msevalues_stdv = np.zeros(num_samples.size, dtype='float64')
            for k, num_samp in enumerate(num_samples):
                print('num_samp: {0!s}'.format(num_samp), flush=True)
                tvalues = np.zeros((num_reps, subds.shape[1]), dtype='float64')
                pvalues = np.zeros((num_reps, subds.shape[1]), dtype='float64')
                pc1values = np.zeros((num_reps, subds.shape[1]), dtype='float64')
                msevalues = np.zeros(num_reps, dtype='float64')
                for rep in range(num_reps):
                    print('rep: {0!s}'.format(rep), flush=True)
                    sidxsi = np.random.choice(idxsi, num_samp, replace=False) # bootstrap?
                    sidxsj = np.random.choice(idxsj, num_samp, replace=False) # bootstrap?
                    sidxs = np.append(sidxsi, sidxsj)
                    istrain = np.in1d(np.arange(subds.shape[0]), sidxs)
                    train = copy.deepcopy(subds)
                    train.discard(~istrain, 0)
                    test = copy.deepcopy(subds)
                    test.discard(istrain, 0)
                    pca_model = PCA().fit(train.matrix)
                    pca_test = pca_model.transform(test.matrix)
                    recon_test = pca_model.inverse_transform(pca_test)
                    msevalues[rep] = ((recon_test - test.matrix)**2).mean()
                    tvalues[rep,:], pvalues[rep,:] = stats.ttest_ind(subds.matrix[sidxsi,:], subds.matrix[sidxsj,:], axis=0, equal_var=False, nan_policy='propagate')
                    pc1values[rep,:] = pca_model.components_[0,:]
                tvalues_mean[k,:] = tvalues.mean(0)
                tvalues_stdv[k,:] = tvalues.std(0)
                pvalues_mean[k,:] = pvalues.mean(0)
                pvalues_stdv[k,:] = pvalues.std(0)
                pc1values_mean[k,:] = pc1values.mean(0)
                pc1values_stdv[k,:] = pc1values.std(0)
                msevalues_mean[k] = msevalues.mean()
                msevalues_stdv[k] = msevalues.std()
                print('num_samp: {0!s}, mse_mean: {1:1.3g}, mse_stdv: {2:1.3g}'.format(num_samp, msevalues_mean[k], msevalues_stdv[k]), flush=True)
                # need to collect the "true" tvalues and pvalues and pc1values and msevalue
                # need to compute fraction significant from pvalues, not pvalue mean and stdv
                # get ranks and look at mean and stdv of ranks
                
                    

            
pca_model = PCA(n_components=2).fit(snp_genome.matrix)
pca_matrix = pca_model.transform(snp_genome.matrix)


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
                genotype_matrix.append([float(x.strip()) for x in genotypes.replace('0|0','0').replace('0|1','0.5').replace('1|0','0.5').replace('1|1','1').split('\t')])
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
                                       columnname='genome_id',
                                       columnlabels=np.array(genome_ids, dtype='object'),
                                       columnmeta={'population':np.array(pops, dtype='object'), 'super_population':np.array(super_pops, dtype='object'), 'gender':np.array(genders, dtype='object')},
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

# add SNP mean and stdv to metadata
print('adding SNP mean and stdv to metadata...', flush=True)
snp_genome.rowmeta['mean'] = snp_genome.matrix.mean(1) 
snp_genome.rowmeta['stdv'] = snp_genome.matrix.std(1)

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
savefolder = '../../input_data/1000genomes_genomes'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, snp_genome)
shutil.copyfile('../../original_data/1000genomes/snp_genome_1000genomes-phased-MHC_prepared.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/1000genomes/snp_genome_1000genomes-phased-MHC_prepared.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)


# visualization
pca_model = PCA(n_components=2).fit(snp_genome.matrix)
pca_matrix = pca_model.transform(snp_genome.matrix)
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.plot(pca_matrix[:,0], pca_matrix[:,1], 'ok', markersize=1, markeredgewidth=0, alpha=0.5, zorder=0)
ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/snp_genome_pca.png', transparent=True, pad_inches=0, dpi=300)
plt.close()
for metalabel in ['mean', 'stdv', 'position']:
    z = snp_genome.rowmeta[metalabel].astype('float64')
    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
    ax.scatter(pca_matrix[:,0], pca_matrix[:,1],  s=1, c=z, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), alpha=0.5, vmin=z.min(), vmax=z.max())
    ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
    ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
    ax.set_frame_on(False)
    fg.savefig('../../original_data/1000genomes/snp_genome_pca_colored_by_{0}.png'.format(metalabel), transparent=True, pad_inches=0, dpi=300)
    plt.close()
hla_hit = np.array(['HLA-' in x for x in snp_genome.rowmeta['gene_name']], dtype='bool')
hla_names = snp_genome.rowmeta['gene_name'].copy()
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
fg.savefig('../../original_data/1000genomes/snp_genome_pca_colored_by_hlagene.png', transparent=True, pad_inches=0, dpi=300)
plt.close()

pca_model = PCA(n_components=2).fit(snp_genome.matrix.T)
pca_matrix = pca_model.transform(snp_genome.matrix.T)
fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
ax.plot(pca_matrix[:,0], pca_matrix[:,1], 'ok', markersize=2, markeredgewidth=0, alpha=0.5, zorder=0)
ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
ax.set_frame_on(False)
fg.savefig('../../original_data/1000genomes/genome_snp_pca.png', transparent=True, pad_inches=0, dpi=300)
plt.close()
for metalabel in ['population', 'super_population', 'gender']:
    categories = np.unique(snp_genome.columnmeta[metalabel])
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(float((i+0.5)/len(categories))) for i in range(len(categories))]
    fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
    ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
    for category, color in zip(categories, colors):
        hit = snp_genome.columnmeta[metalabel] == category
        ax.plot(pca_matrix[hit,0], pca_matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=0.5, zorder=0, label=category)
    ax.tick_params(axis='both', which='major', bottom=False, top=False, labelbottom=False, labeltop=False, left=False, right=False, labelleft=False, labelright=False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=2, fontsize=8, labelspacing=0.25)
    ax.set_title('expl_var_frac: {0:1.3g}'.format(pca_model.explained_variance_ratio_.sum()), fontsize=8)
    ax.set_frame_on(False)
    fg.savefig('../../original_data/1000genomes/genome_snp_pca_colored_by_{0}.png'.format(metalabel), transparent=True, pad_inches=0, dpi=300)
    plt.close()
