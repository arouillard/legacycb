# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import numpy as np
import datasetIO


# load gtex tissues
print('loading gtex tissues...')
tissues = []
with open('gtex_tissues.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    fr.readline()
    for line in fr:
        tissues.append([x.strip() for x in line.split('\t')])
        print(tissues[-1])

# load termite tissues (just need the metadata, not the intersection counts)
# this file is generated by count_term-term_pmids_from_termite.py
print('loading termite tissues...')
term_term = datasetIO.load_datamatrix('HUCELLANAT_HUCELLANAT_datamatrix_pmidcounts_year_all_datestamp_all_minscore_2.pickle')
print(term_term)

# get number of words in each termite tissue name
word_counts = np.array([len(x.split(' ')) for x in term_term.rowmeta['term_name']], dtype='float64')

# match gtex tissues to termite tissues
print('matching gtex tissues to termite tissues...')
# crude algorithm based on word overlap:
# for each gtex tissue,
#     compute jaccard similarity with each termite tissue (word intersection/word union)
#     sort termite tissues by decreasing jaccard similarity
#     write terms with non-zero jaccard similarity to file
with open('gtex_to_termite_tissues_candidate_mappings.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    fw.write('\t'.join(['gtex_general_tissue', 'gtex_specific_tissue', 'ng3981_tissue', 'candidate', 'jaccard', 'term_count_intersectionunion', 'term_count_union']) + '\n')
    for gtex_general_tissue, gtex_specific_tissue, ng3981_tissue in tissues:
        print('working on gtex tissue: {0}...'.format(ng3981_tissue))
        tissue_words = ng3981_tissue.lower().split('_') # words in gtex tissue name
        hit_arrays = []
        for tissue_word in tissue_words:
            hit_arrays.append([tissue_word in x.lower() for x in term_term.rowmeta['term_name']]) # check if gtex tissue word is in each termite tissue name
        hit_arrays = np.array(hit_arrays, dtype='bool') # rows=words in gtex tissue, cols=termite tissues
        hit_counts = hit_arrays.sum(0) # number of words in gtex tissue and in termite tissue, for each termite tissue
        jaccard = hit_counts/(word_counts + len(tissue_words) - hit_counts) # word intersection/word union
        si = np.argsort(jaccard)[::-1]
        for candidate, score, term_count_intersectionunion, term_count_union in zip(term_term.rowlabels[si], jaccard[si], term_term.rowmeta['term_count_intersectionunion'][si], term_term.rowmeta['term_count_union'][si]):
            if score == 0:
                break
            fw.write('\t'.join([gtex_general_tissue, gtex_specific_tissue, ng3981_tissue, candidate, '{0:1.3g}'.format(score), str(term_count_intersectionunion), str(term_count_union)]) + '\n')

print('done get_gtex_to_termite_tissue_candidate_mapping.py')
