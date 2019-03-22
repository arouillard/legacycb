# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import argparse
import gzip
import os
import pickle
from collections import defaultdict
import numpy as np


def main(dictionary, year, datestamp, min_score):
    
    print('dictionary: {0}'.format(dictionary))
    print('year: {0}'.format(year))
    print('datestamp: {0}'.format(datestamp))
    print('min_score: {0!s}'.format(min_score))
    
    # specify location of flat files containing term-publication associations discovered by Termite
    termite_path = '/GWD/bioinfo/projects/RD-TSci-CommonData/external/medline/termite_tsv'
    print('termite_path: {0}'.format(termite_path))
    
    # finding year folders to search
    print('finding year folders to search for files...')
    years = [int(x) for x in os.listdir(termite_path)]
    if year == 'latest':
        years = [max(years)]
    elif year == 'all':
        pass
    else:
        years = [int(year)]
    print('years', years)
    
    # finding files to parse
    print('finding files to parse...')
    data_paths = []
    for yr in years:
        dictionary_path = '{0}/{1!s}/{2}'.format(termite_path, yr, dictionary)
        filenames = os.listdir(dictionary_path)
        filenames = [x for x in filenames if 'TER-medline' in x and '_all.tsv.gz' in x]
        datestamps = np.array([x.replace('TER-medline', '').replace('_all.tsv.gz','') for x in filenames], dtype='int64')
        if datestamp == 'latest':
            hidx = np.argmax(datestamps)
            filename = filenames[hidx]
            path = '{0}/{1}'.format(dictionary_path, filename)
            data_paths.append(path)
        elif datestamp == 'all':
            data_paths += ['{0}/{1}'.format(dictionary_path, filename) for filename in filenames]
        else:
            hidx = np.where(datestamps == int(datestamp))[0][0]
            filename = filenames[hidx]
            path = '{0}/{1}'.format(dictionary_path, filename)
            data_paths.append(path)
    print('\n'.join(data_paths), flush=True)

    # create a dict of PMIDs for each term
    # terms are uniquely identified as dictionary$termid$termname
    # consider only term-publication associations with Termite score > 1
    # (if Termite score < 2 then term likely is irrelevant to publication)
    print('parsing files...')
    term_pmids = defaultdict(set) # dict containing set of PMIDs for each term
    score_count = defaultdict(int) # dict containing counts of termite scores
    for path in data_paths:
        print('working on {0}'.format(path), flush=True)
        with gzip.open(path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            for i in range(4):
                fr.readline()
            for line in fr:
                entries = [x.strip() for x in line.split('\t', maxsplit=8)]
                if line[0] == '#' or len(entries) != 9:
                    continue
                else:
                    pmid, termid, termname = entries[4:7]
                    score = int(entries[7])
                    score_count[score] += 1
                    if score >= min_score:
                        term_pmids['$'.join([dictionary, termid, termname])].add(pmid) 

    # save results
    print('saving results...')
    with open('term_pmid_dict_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.pickle'.format(dictionary, year, datestamp, min_score), 'wb') as fw:
        pickle.dump(term_pmids, fw)
    with open('score_count_dict_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.pickle'.format(dictionary, year, datestamp, min_score), 'wb') as fw:
        pickle.dump(score_count, fw)
    
    # write pmid counts to text file
    with open('term_pmid_counts_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.txt'.format(dictionary, year, datestamp, min_score), 'wt') as fw:
        fw.write('\t'.join(['dictionary', 'termid', 'termname', 'pmid_count']) + '\n')
        for term, pmids in term_pmids.items():
            dictionary, termid, termname = term.split('$')
            fw.write('\t'.join([dictionary, termid, termname, str(len(pmids))]) + '\n')
    
    # print distribution of Termite scores
    print('distribution of Termite scores:')
    for score, count in score_count.items():
        print('{0!s},{1!s}'.format(score, count), flush=True)
    
    print('done get_term_pmids_from_termite.py', flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get term pmids from termite.')
    parser.add_argument('--dictionary', help='dictionary of biomedical terms', type=str)
    parser.add_argument('--year', help='year of termite file', type=str, default='all')
    parser.add_argument('--datestamp', help='datestamp of termite file', type=str, default='all')
    parser.add_argument('--min_score', help='min score for filtering term-pmid associations', type=int, default=2)
    args = parser.parse_args()
    main(args.dictionary, args.year, args.datestamp, args.min_score)
