# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biology
GSK
"""

import pandas as pd
import os

print('loading dataframe...', flush=True)
df = pd.read_csv('../../original_data/qtq/combined_feature_OTv8_all_indication.csv.gz')

features = ['expression_atlas', 'uniprot', 'gwas_catalog', 'eva', 'uniprot_literature', 'gene2phenotype', 'reactome', 'phenodigm', 'cancer_gene_census', 'eva_somatic', 'intogen', 'chembl', 'europepmc', 'metabase', 'pharmaprojects', 'targetpedia', 'arrayserver', 'GeneLogic_overexpression', 'GTEX_overexpression', 'STOPGAP251_GWAS', 'TERMITE_literature', 'Uhlen_overexpression', 'ExAC_LoF', 'ExAC_Missense', 'GTEX_median_all_tissues', 'Mouse_Protein_Identity', 'Rare_Variant_Intolerance_Score', 'Target_Location', 'Target_Topology']

for i, feature in enumerate(features):
    
    print('working on feature {0!s} of {1!s}: {2}'.format(i+1, len(features), feature), flush=True)
    
    os.makedirs('../../original_data/{0}'.format(feature))
   
    gd = df.pivot(index='EntrezGeneID', columns='MeSH ID', values=feature)
    
    gd.to_csv('../../original_data/{0}/geneid_meshid_datamatrix_untrimmed.csv'.format(feature))
    
    gd.dropna(axis=1, how='all', inplace=True)
    
    gd.dropna(axis=0, how='all', inplace=True)
    
    gd.fillna(value=0, inplace=True)
    
    gd.to_csv('../../original_data/{0}/geneid_meshid_datamatrix_trimmed.csv'.format(feature))

    del gd
