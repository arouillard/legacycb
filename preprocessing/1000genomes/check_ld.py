# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import datasetIO

# load prepared data
print('loading prepared data...', flush=True)
snp_genome = datasetIO.load_datamatrix('../../original_data/1000genomes/snp_genome_1000genomes-phased-MHC_prepared.pickle')
print(snp_genome, flush=True)

# filter SNPs
print('filtering SNPs...', flush=True)
#selected_snps = ['rs35651056', 'rs4713167', 'rs570380504', 'rs539402503', 'rs142905864', 'rs182148819', 'rs59393499', 'rs112696801', 'rs543963198', 'rs562658944']
with open('../../original_data/1000genomes/first_990_rsids.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    selected_snps = [x.strip() for x in fr.read().split('\n')[:20]]
tobediscarded = ~np.in1d(snp_genome.rowlabels, selected_snps)
snp_genome.discard(tobediscarded, 0)
print(snp_genome, flush=True)

# calculate pearson correlations for each population
print('calculating pearson correlations for each population...', flush=True)
with open('../../original_data/1000genomes/snp_snp_checkLD.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    fw.write('\t'.join(['category_name', 'category', 'SNP_A', 'SNP_B', 'r', 'r_squared', 'D', 'Dprime', 'r', 'r_squared']) + '\n')
    for category_name in ['population', 'super_population']:
        print('    category_name: {0}'.format(category_name), flush=True)
        categories = np.unique(snp_genome.columnmeta[category_name])
        for category in categories:
            print('    category: {0}'.format(category), flush=True)
            hit = snp_genome.columnmeta[category_name] == category
            sg = snp_genome.tolabels(columnlabels=snp_genome.columnlabels[hit])
            pa = sg.matrix.sum(1, keepdims=True)/sg.shape[1]
            pb = pa.T
            pab = sg.matrix.dot(sg.matrix.T)/sg.shape[1]
            D = pab - pa*pb
            Dmaxneg = np.max(np.concatenate(((-pa*pb)[:,:,np.newaxis], (-(1-pa)*(1-pb))[:,:,np.newaxis]), 2), 2)
            Dmaxpos = np.min(np.concatenate(((pa*(1-pb))[:,:,np.newaxis], ((1-pa)*pb)[:,:,np.newaxis]), 2), 2)
            Dmax = np.zeros(D.shape, dtype='float32') + (D < 0)*Dmaxneg + (D > 0)*Dmaxpos
            Dprime = D/Dmax
            r = D/np.sqrt(pa*(1-pa)*pb*(1-pb))
            rsquared = r**2
            snp_snp = sg.tosimilarity(0, 'pearson')
#            print(snp_snp.rowlabels, flush=True)
#            print(snp_snp.matrix, flush=True)
#            I, J = np.logical_and(np.triu(np.ones(snp_snp.shape, dtype='bool'), 1), ~np.isnan(snp_snp.matrix)).nonzero()
            I, J = (~np.isnan(snp_snp.matrix)).nonzero()
            print('\t'.join(['category_name', 'category', 'SNP_A', 'SNP_B', 'r', 'r_squared', 'D', 'Dprime', 'r', 'r_squared']), flush=True)
            for i,j in zip(I, J):
                print('\t'.join([category_name, category, snp_snp.rowlabels[i], snp_snp.columnlabels[j], str(snp_snp.matrix[i,j]), str(snp_snp.matrix[i,j]**2), str(D[i,j]), str(Dprime[i,j]), str(r[i,j]), str(rsquared[i,j])]), flush=True)
                fw.write('\t'.join([category_name, category, snp_snp.rowlabels[i], snp_snp.columnlabels[j], str(snp_snp.matrix[i,j]), str(snp_snp.matrix[i,j]**2), str(D[i,j]), str(Dprime[i,j]), str(r[i,j]), str(rsquared[i,j])]) + '\n')
#            datasetIO.save_datamatrix('../../original_data/1000genomes/snp_snp_checkLD_{0}_{1}.txt.gz'.format(category_name, category), snp_snp)

print('done.')
