# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

import sys
sys.path.append('../../utilities')

import numpy as np
import datasetIO
import os

# load the data
print('loading datamatrix 1...', flush=True)
dm1_name = 'GTEXv6'
dm1_likelihood = 'normal'
dm1 = datasetIO.load_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_scaledcounts_prepared.pickle')
dm1.rowmeta['row_mean_ref'] = dm1.rowmeta['row_mean_ref'].reshape(-1) # must be 1d to be compatible with tolabels method later
dm1.rowmeta['row_stdv_ref'] = dm1.rowmeta['row_stdv_ref'].reshape(-1) # must be 1d to be compatible with tolabels method later
print(dm1, flush=True)
print('loading datamatrix 2...', flush=True)
dm2_name = 'impc'
dm2_likelihood = 'bernoulli'
dm2 = datasetIO.load_datamatrix('../../original_data/impc/gene_phenotype_impc_trimmed_thresholded_propagated_prepared.pickle')
print(dm2, flush=True)

# align rows
print('aligning rows...', flush=True)
u_rowlabels = np.union1d(dm1.rowlabels, dm2.rowlabels)
i_rowlabels = np.intersect1d(dm1.rowlabels, dm2.rowlabels)
in_dm1 = np.in1d(u_rowlabels, dm1.rowlabels)
in_dm2 = np.in1d(u_rowlabels, dm2.rowlabels)
in_dm1_only = np.logical_and(in_dm1, ~in_dm2)
in_dm2_only = np.logical_and(in_dm2, ~in_dm1)
i_rowmetafields = set(list(dm1.rowmeta.keys())).intersection(list(dm2.rowmeta.keys()))
print('row counts dm1:{0!s}, dm2:{1!s}, intersection:{2!s}, union:{3!s}'.format(dm1.shape[0], dm2.shape[0], i_rowlabels.size, u_rowlabels.size), flush=True)
dm1 = dm1.tolabels(rowlabels=u_rowlabels.copy(), fillvalue=np.nan)
print(dm1, flush=True)
dm2 = dm2.tolabels(rowlabels=u_rowlabels.copy(), fillvalue=np.nan)
print(dm2, flush=True)
for field in i_rowmetafields:
    dm1.rowmeta[field][in_dm2_only] = dm2.rowmeta[field][in_dm2_only]
    dm2.rowmeta[field][in_dm1_only] = dm1.rowmeta[field][in_dm1_only]

# annotate columns
print('annotating columns...', flush=True)
dm1.columnmeta['dataset'] = np.full(dm1.shape[1], dm1_name, dtype='object')
dm1.columnmeta['likelihood'] = np.full(dm1.shape[1], dm1_likelihood, dtype='object')
dm1.columnmeta[dm1.columnname] = dm1.columnlabels.copy()
dm1.columnmeta['feature'] = dm1.columnlabels.copy()
dm1.columnname = 'feature|dataset'
dm1.columnlabels = dm1.columnmeta['feature'] + '|' + dm1.columnmeta['dataset']

dm2.columnmeta['dataset'] = np.full(dm2.shape[1], dm2_name, dtype='object')
dm2.columnmeta['likelihood'] = np.full(dm2.shape[1], dm2_likelihood, dtype='object')
dm2.columnmeta[dm2.columnname] = dm2.columnlabels.copy()
dm2.columnmeta['feature'] = dm2.columnlabels.copy()
dm2.columnname = 'feature|dataset'
dm2.columnlabels = dm2.columnmeta['feature'] + '|' + dm2.columnmeta['dataset']

# merge datasets
print('merging datasets...', flush=True)
dm = dm1.concatenate(dm2, 'self', 1)
dm.rowmeta['in_dm1'] = in_dm1.copy()
dm.rowmeta['in_' + dm1_name] = in_dm1.copy()
dm.rowmeta['in_dm2'] = in_dm2.copy()
dm.rowmeta['in_' + dm2_name] = in_dm2.copy()
dm.columnmeta['in_dm1'] = dm.columnmeta['dataset'] == dm1_name
dm.columnmeta['in_' + dm1_name] = dm.columnmeta['dataset'] == dm1_name
dm.columnmeta['in_dm2'] = dm.columnmeta['dataset'] == dm2_name
dm.columnmeta['in_' + dm2_name] = dm.columnmeta['dataset'] == dm2_name
dm.matrixname = dm1_name + '_' + dm2_name + '_merged'
print(dm, flush=True)
print(dm.rowmeta.keys(), flush=True)
print(dm.columnmeta.keys(), flush=True)

# save the data
print('saving merged data...', flush=True)
savefolder = '../../input_data/{0}_{1}'.format(dm1_name, dm2_name)
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
datasetIO.save_splitdata(savefolder, dm)
datasetIO.save_datamatrix('{0}/datamatrix.pickle'.format(savefolder), dm)
datasetIO.save_datamatrix('{0}/datamatrix.txt.gz'.format(savefolder), dm)

print('done.', flush=True)
