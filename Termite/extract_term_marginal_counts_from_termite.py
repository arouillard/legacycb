# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import argparse
import datasetIO

def main(dictionary, year, datestamp, min_score):
    
    print('begin extract_term_marginal_counts_from_termite.py')
    
    print('dictionary: {0}'.format(dictionary))
    print('year: {0}'.format(year))
    print('datestamp: {0}'.format(datestamp))
    print('min_score: {0!s}'.format(min_score))
    
    # load counts datamatrix
    # this file is generated by count_term-term_pmids_from_termite.py    
    print('loading counts datamatrix...')
    row_dictionary = dictionary # 'HUCELL', 'ANAT', 'INDICATION', 'HUCELLANAT', 'HUCELLANATINDICATION'
    column_dictionary = dictionary # 'HUCELL', 'ANAT', 'INDICATION', 'HUCELLANAT', 'HUCELLANATINDICATION'
    counts_datamatrix_path = '{0}_{1}_datamatrix_pmidcounts_year_{2}_datestamp_{3}_minscore_{4!s}.pickle'.format(row_dictionary, column_dictionary, year, datestamp, min_score)
    term_term = datasetIO.load_datamatrix(counts_datamatrix_path)
    print('counts_datamatrix_path: {0}'.format(counts_datamatrix_path))
    print(term_term)
    
    # write marginal counts to file
    print('writing marginal counts...')
    metalabels = sorted(list(term_term.rowmeta.keys()))
    with open('{0}_term_marginal_pmidcounts_year_{1}_datestamp_{2}_minscore_{3!s}.txt'.format(dictionary, year, datestamp, min_score), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        writelist = [term_term.rowname] + metalabels
        fw.write('\t'.join(writelist) + '\n')
        for i, rowlabel in enumerate(term_term.rowlabels):
            writelist = [rowlabel] + [term_term.rowmeta[k][i] if term_term.rowmeta[k].dtype == 'object' else str(term_term.rowmeta[k][i]) for k in metalabels]
            fw.write('\t'.join(writelist) + '\n')
    
    print('done extract_term_marginal_counts_from_termite.py')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get marginal counts from termite.')
    parser.add_argument('--dictionary', help='dictionary of biomedical terms', type=str)
    parser.add_argument('--year', help='year of termite file', type=str, default='all')
    parser.add_argument('--datestamp', help='datestamp of termite file', type=str, default='all')
    parser.add_argument('--min_score', help='min score for filtering term-pmid associations', type=int, default=2)
    args = parser.parse_args()
    main(args.dictionary, args.year, args.datestamp, args.min_score)
