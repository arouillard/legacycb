# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import pandas as pd
import copy
import datasetIO
import os
import shutil
from dataclasses import datamatrix as DataMatrix


# load the data
print('loading dataset...', flush=True)
dataset = datasetIO.load_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical.pickle')
print(dataset, flush=True)

# discard samples
print('discarding samples...', flush=True)
dataset.discard(dataset.rowmeta['irrecist'] == 'stable disease', 0)
print(dataset, flush=True)

# save the data
print('saving data...', flush=True)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_no_stabledisease.pickle', dataset)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_no_stabledisease.txt.gz', dataset)
savefolder = '../../input_data/pratfelip_transposed_plus_clinical_no_stabledisease'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, dataset)
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_no_stabledisease.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_gene_pratfelip_nanostring_prepared_plus_clinical_no_stabledisease.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

# load the data
print('loading dataset...', flush=True)
dataset = datasetIO.load_datamatrix('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv.pickle')
print(dataset, flush=True)

# discard samples
print('discarding samples...', flush=True)
dataset.discard(dataset.rowmeta['irrecist'] == 'stable disease', 0)
print(dataset, flush=True)

# save the data
print('saving data...', flush=True)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv_no_stabledisease.pickle', dataset)
datasetIO.save_datamatrix('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv_no_stabledisease.txt.gz', dataset)
savefolder = '../../input_data/pratfelip_clinical_and_deconv_no_stabledisease'
if not os.path.exists(savefolder):
	os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, dataset)
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv_no_stabledisease.pickle', '{0}/datamatrix.pickle'.format(savefolder))
shutil.copyfile('../../original_data/pratfelip_symlnk/patient_ft_pratfelip_only_clinical_and_deconv_no_stabledisease.txt.gz', '{0}/datamatrix.txt.gz'.format(savefolder))

print('done.', flush=True)
