# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import copy
import datasetIO
import os
import shutil
from collections import defaultdict


# load the data
print('loading dataset...', flush=True)
dataset = datasetIO.load_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared.pickle')

rgep_path = {'pfprior':'../../original_data/rgeps_symlnk/PratFelip_RGEPs.txt',
             'pfleak':'../../original_data/rgeps_symlnk/PratFelip_information_leak_RGEPs.txt',
             'knowledge_based':'../../original_data/rgeps_symlnk/GSK_knowledge_based_RGEPs.txt',
             'nanostring':'../../original_data/rgeps_symlnk/Nanostring_RGEPs.txt',
             'melanoma_single_cell':'../../original_data/rgeps_symlnk/TiroshGarraway_melanoma_single_cell_RGEPs.txt'}

for rgep_name, rgep_path in rgep_path.items():
    
    # load rgep
    print('working on rgep: {0}...'.format(rgep_name), flush=True)
    print('loading rgep...', flush=True)
    gene_cell = defaultdict(set)
    with open(rgep_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        for line in fr:
            gene_sym, cell_type = [x.strip() for x in line.split('\t')]
            gene_cell[gene_sym].add(cell_type)
    for gene_sym, cell_type in gene_cell.items():
        gene_cell[gene_sym] = '|'.join(sorted(cell_type))
    
    # copy dataset
    print('copying dataset...', flush=True)
    atb_gene = copy.deepcopy(dataset)
    
    # filter genes
    print('filtering genes...', flush=True)
    tobediscarded = ~np.in1d(atb_gene.columnmeta['symbol'], list(gene_cell.keys()))
    atb_gene.discard(tobediscarded, 1)
    atb_gene.matrixname += '_filtered_by_{0}_rgep'.format(rgep_name)
    print('rgep_genes: {0!s}'.format(len(gene_cell)), flush=True)
    print(atb_gene)
    
    # add cell type metadata
    print('adding cell type metadata...', flush=True)
    atb_gene.columnmeta['rgep_cell_type'] = np.array([gene_cell[gene_sym] for gene_sym in atb_gene.columnmeta['symbol']], dtype='object')
        
    # save the data
    print('saving filtered data...', flush=True)
    datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_filtered_by_{0}_rgep.pickle'.format(rgep_name), atb_gene)
    datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_filtered_by_{0}_rgep.txt.gz'.format(rgep_name), atb_gene)
    savefolder = '../../input_data/hugolo_transposed_filtered_by_{0}_rgep'.format(rgep_name)
    if not os.path.exists(savefolder):
    	os.makedirs(savefolder)
    datasetIO.save_splitdata(savefolder, atb_gene)
    shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_filtered_by_{0}_rgep.pickle'.format(rgep_name), '{0}/datamatrix.pickle'.format(savefolder))
    shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_filtered_by_{0}_rgep.txt.gz'.format(rgep_name), '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
