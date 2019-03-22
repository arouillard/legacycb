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
from dataclasses import datamatrix as DataMatrix


# load the data
print('loading dataset...', flush=True)
dataset = datasetIO.load_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared.pickle')
dataset.columnmeta[dataset.columnname] = dataset.columnlabels.copy()
dataset.columnname += '_or_clinical_variable'
dataset.columnmeta['variable_type'] = np.full(dataset.shape[1], 'gene', dtype='object')
dataset.columnmeta['is_gene'] = np.ones(dataset.shape[1], dtype='bool')
dataset.columnmeta['is_clinical_variable'] = np.zeros(dataset.shape[1], dtype='bool')
print(dataset, flush=True)

# create datamatrix of clinical variables
print('creating datamatrix of clinical variables...', flush=True)
clinical_variables = ['study_site_ucla', 'gender_male', 'age_45to55', 'age_57to63', 'age_65to74', 'age_82to84', 'disease_status_m1c', 'previous_mapk_inhibitor', 'anatomical_site_arm', 'anatomical_site_headneck', 'anatomical_site_leg', 'anatomical_site_lung', 'anatomical_site_torso', 'braf_mutation', 'nras_mutation', 'nf1_mutation']
clinical_dataset = DataMatrix(rowname=dataset.rowname,
                              rowlabels=dataset.rowlabels.copy(),
                              rowmeta=copy.deepcopy(dataset.rowmeta),
                              columnname=dataset.columnname,
                              columnlabels=clinical_variables.copy(),
                              columnmeta={'variable_type':np.full(len(clinical_variables), 'clinical_variable', dtype='object'), 'is_gene':np.zeros(len(clinical_variables), dtype='bool'), 'is_clinical_variable':np.ones(len(clinical_variables), dtype='bool')},
                              matrixname='clinical_variables_for_tumor_samples',
                              matrix=np.concatenate(tuple(dataset.rowmeta[cv].reshape(-1,1) for cv in clinical_variables), 1).astype('float64'))
print(clinical_dataset, flush=True)

# append clinical variables
print('appending clinical variables...', flush=True)
dataset.append(clinical_dataset, 1)
dataset.matrixname += '_and_clinical_variables'
print(dataset, flush=True)

# save the data
print('saving data with clinical variables...', flush=True)
datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical.pickle', dataset)
datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical.txt.gz', dataset)
savefolder = '../../input_data/hugolo_transposed_plus_clinical'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, dataset)
shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

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
    
    # copy dataset
    print('copying dataset...', flush=True)
    pt_var = copy.deepcopy(dataset)
    
    # filter genes
    print('filtering genes...', flush=True)
    tobediscarded = np.logical_and(pt_var.columnmeta['is_gene'], ~np.in1d(pt_var.columnmeta['symbol'], list(gene_cell.keys())))
    pt_var.discard(tobediscarded, 1)
    pt_var.matrixname += '_filtered_by_{0}_rgep'.format(rgep_name)
    print('rgep_genes: {0!s}'.format(len(gene_cell)), flush=True)
    print(pt_var)
    
    # add cell type metadata
    print('adding cell type metadata...', flush=True)
    pt_var.columnmeta['rgep_cell_type'] = np.array(['|'.join(sorted(gene_cell[gene_sym])) for gene_sym in pt_var.columnmeta['symbol']], dtype='object')
        
    # save the data
    print('saving filtered data...', flush=True)
    datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical_filtered_by_{0}_rgep.pickle'.format(rgep_name), pt_var)
    datasetIO.save_datamatrix('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical_filtered_by_{0}_rgep.txt.gz'.format(rgep_name), pt_var)
    savefolder = '../../input_data/hugolo_transposed_plus_clinical_filtered_by_{0}_rgep'.format(rgep_name)
    if not os.path.exists(savefolder):
    	os.makedirs(savefolder)
    datasetIO.save_splitdata(savefolder, pt_var)
    shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical_filtered_by_{0}_rgep.pickle'.format(rgep_name), '{0}/datamatrix.pickle'.format(savefolder))
    shutil.copyfile('../../original_data/hugolo_symlnk/patient_gene_hugolo_rnaseq_prepared_plus_clinical_filtered_by_{0}_rgep.txt.gz'.format(rgep_name), '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
