# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biology
GSK
"""

import pandas as pd

print('loading dataframe...', flush=True)
df = pd.read_csv('../../original_data/qtq/combined_feature_OTv8_all_indication.csv.gz')

features = ['expression_atlas', 'uniprot', 'gwas_catalog', 'eva', 'uniprot_literature', 'gene2phenotype', 'reactome', 'phenodigm', 'cancer_gene_census', 'eva_somatic', 'intogen', 'chembl', 'europepmc', 'metabase', 'pharmaprojects', 'targetpedia', 'arrayserver', 'GeneLogic_overexpression', 'GTEX_overexpression', 'STOPGAP251_GWAS', 'TERMITE_literature', 'Uhlen_overexpression', 'ExAC_LoF', 'ExAC_Missense', 'GTEX_median_all_tissues', 'Mouse_Protein_Identity', 'Rare_Variant_Intolerance_Score', 'Target_Location', 'Target_Topology']

feature = 'phenodigm'
    
print('working on feature: {0}'.format(feature), flush=True)
    
gd = df.pivot(index='EntrezGeneID', columns='MeSH ID', values=feature)
gd.to_csv('../../original_data/{0}/geneid_meshid_datamatrix_untrimmed.csv.gz'.format(feature), compression='gzip')
gd.dropna(axis=1, how='all', inplace=True)
gd.dropna(axis=0, how='all', inplace=True)
gd.fillna(value=0, inplace=True)
gd.to_csv('../../original_data/{0}/geneid_meshid_datamatrix_trimmed.csv.gz'.format(feature), compression='gzip')

