# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import pandas as pd

def annotate_genes(field, values, metadatapath, fields=['gene_family', 'locus_type', 'gene_family_id', 'hgnc_id', 'ensembl_gene_id', 'entrez_id', 'symbol', 'name', 'uniprot_ids'], drop_duplicates=False):    
    if type(fields) == list and len(fields) > 0:
        if field not in fields:
            fields = [field] + fields
        df = pd.read_table(metadatapath, index_col=False, usecols=fields, dtype=str)
    else:
        df = pd.read_table(metadatapath, index_col=False, dtype=str)
    if field not in df.columns:
        raise ValueError('field {0} is not in metadata'.format(field))
    else:
        df.dropna(axis=0, how='any', subset=[field], inplace=True)
        if drop_duplicates:
            num_rows_start = df.shape[0]
            df.drop_duplicates(subset=[field], keep=False, inplace=True)
            num_rows_dup = num_rows_start - df.shape[0]
            if num_rows_dup > 0:
                print('WARNING! DROPPED {0!s} DUPLICATE ROWS IN METADATA FILE.'.format(num_rows_dup), flush=True)
        values_as_df = pd.DataFrame(values, index=None, columns=[field])
        df_aligned_to_values = pd.merge(values_as_df, df, how='left', on=field, sort=False, validate='many_to_one')
        df_aligned_to_values.fillna(value='nan', inplace=True)
        return {f:df_aligned_to_values[f].values for f in df_aligned_to_values.columns}
