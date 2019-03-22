# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biology
GSK
"""

import pandas as pd
import numpy as np

print('loading dataframe...', flush=True)
df = pd.read_csv('../../original_data/impc/IMPC_genotype_phenotype.csv.gz') # (33970, 28)
print(df, flush=True)

print('discarding rows missing gene or phenotype identifier...', flush=True)
df.dropna(axis=0, how='any', subset=['marker_accession_id', 'mp_term_id'], inplace=True) # (33749, 28)
print(df.shape, flush=True)

#print('discarding rows missing summary stats (p_value, percentage_change, effect_size)...', flush=True)
#df.dropna(axis=0, how='any', subset=['p_value', 'percentage_change', 'effect_size'], inplace=True) # (21576, 28)
#print(df.shape, flush=True)

print('discarding rows missing p_values...', flush=True)
df.dropna(axis=0, subset=['p_value'], inplace=True) # (29006, 28)
print(df.shape, flush=True)

print('creating gene-phenotype pivot table filled with max p-values...', flush=True)
print('only associations with 1e-4 significance level are in the dataframe.', flush=True)
print('aggregating by max p-value (least significant p-value) is the most conservative thing to do.', flush=True)
gp = df.pivot_table(index='marker_accession_id', columns='mp_term_id', values='p_value', aggfunc=np.max) # (3455, 295)
print(gp, flush=True)

print('dropping empty rows and columns (there should be none)...', flush=True)
gp.dropna(axis=1, how='all', inplace=True)
gp.dropna(axis=0, how='all', inplace=True)
print(gp.shape, flush=True)

print('filling in missing values...', flush=True)
print('missing values are actually nonsignificant associations.', flush=True)
print('fill in as p_value=1 for now, but remember that these values could be anywhere between 1 and the significance cutoff 1e-4', flush=True)
gp.fillna(value=1, inplace=True)
print(gp.shape, flush=True)

print('saving gene-phenotype pivot table filled with p-values...', flush=True)
print('remember entries with p-value=1 are imputed...', flush=True)
gp.to_csv('../../original_data/impc/mousegeneid_mousephenotypeid_datamatrix_trimmed.csv.gz', compression='gzip')

print('done.', flush=True)
