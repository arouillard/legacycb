# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import argparse
import os
import pickle
import copy
import numpy as np
import dataclasses
import datasetIO
from collections import defaultdict

def get_labels_and_metadata(term_pmids):
    '''parse term metadata into an array of term_dictidnames formatted as dictionary$termid$termname
    and a dictionary of arrays term_dict, term_id, and term_name'''
    term_dictidnames = np.array(sorted(list(term_pmids.keys())), dtype='object')
    term_dicts = np.empty(term_dictidnames.size, dtype='object')
    term_ids = np.empty(term_dictidnames.size, dtype='object')
    term_names = np.empty(term_dictidnames.size, dtype='object')
    for i, term_dictidname in enumerate(term_dictidnames):
        term_dicts[i], term_ids[i], term_names[i] = term_dictidname.split('$')
    return term_dictidnames, {'term_id':term_ids, 'term_name':term_names, 'term_dict':term_dicts}

def main(dictionaries, year, datestamp, min_score):
    
    print('dictionaries: {0}, {1}'.format(dictionaries[0], dictionaries[1]))
    print('year: {0}'.format(year))
    print('datestamp: {0}'.format(datestamp))
    print('min_score: {0!s}'.format(min_score))
    
    # set term dictionaries and paths to dicts containing PMIDs for each term
    # these files are generated by get_term_pmids_from_termite.py    
    row_dictionary = dictionaries[0] # 'HUCELL', 'ANAT', 'INDICATION', 'HUCELLANAT', 'HUCELLANATINDICATION'
    row_pmids_path = 'term_pmid_dict_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.pickle'.format(row_dictionary, year, datestamp, min_score)
    column_dictionary = dictionaries[1] # 'HUCELL', 'ANAT', 'INDICATION', 'HUCELLANAT', 'HUCELLANATINDICATION'
    column_pmids_path = 'term_pmid_dict_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.pickle'.format(column_dictionary, year, datestamp, min_score)
    
    hucellanat_path = 'term_pmid_dict_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.pickle'.format('HUCELLANAT', year, datestamp, min_score)
    if 'HUCELLANAT' in dictionaries and not os.path.exists(hucellanat_path):
        # combine HUCELL and ANAT term-pmid dicts into a single dict
        print('creating {0}...'.format(hucellanat_path), flush=True)
        with open(hucellanat_path.replace('ANAT',''), 'rb') as fr:
            term_pmids = pickle.load(fr)
        with open(hucellanat_path.replace('HUCELL',''), 'rb') as fr:
            term_pmids.update(pickle.load(fr))
        with open(hucellanat_path, 'wb') as fw:
            pickle.dump(term_pmids, fw)
        del term_pmids
    
    hucellanatindication_path = 'term_pmid_dict_dictionary_{0}_year_{1}_datestamp_{2}_minscore_{3!s}.pickle'.format('HUCELLANATINDICATION', year, datestamp, min_score)
    if 'HUCELLANATINDICATION' in dictionaries and not os.path.exists(hucellanatindication_path):
        # combine HUCELL ANAT and INDICATION term-pmid dicts into a single dict
        print('creating {0}...'.format(hucellanatindication_path), flush=True)
        with open(hucellanatindication_path.replace('HUCELLANATINDICATION','HUCELL'), 'rb') as fr:
            term_pmids = pickle.load(fr)
        with open(hucellanatindication_path.replace('HUCELLANATINDICATION','ANAT'), 'rb') as fr:
            term_pmids.update(pickle.load(fr))
        with open(hucellanatindication_path.replace('HUCELLANATINDICATION','INDICATION'), 'rb') as fr:
            term_pmids.update(pickle.load(fr))
        with open(hucellanatindication_path, 'wb') as fw:
            pickle.dump(term_pmids, fw)
        del term_pmids
    
    # first dictionary of biomedical terms
    # load dict mapping terms to PMID sets
    # parse dict to rowlabels and rowmetadata
    print('loading row_dictionary: {0}...'.format(row_dictionary), flush=True)
    with open(row_pmids_path, 'rb') as fr:
        rowterm_pmids = pickle.load(fr)
    rowlabels, rowmeta = get_labels_and_metadata(rowterm_pmids)
    
    # second dictionary of biomedical terms
    # load dict mapping terms to PMID sets
    # parse dict to columnlabels and columnmetadata
    print('loading column_dictionary: {0}...'.format(column_dictionary), flush=True)
    if column_dictionary == row_dictionary:
        columnterm_pmids = rowterm_pmids
        columnlabels = rowlabels
        columnmeta = rowmeta
    else:
        with open(column_pmids_path, 'rb') as fr:
            columnterm_pmids = pickle.load(fr)
        columnlabels, columnmeta = get_labels_and_metadata(columnterm_pmids)
    
    # create datamatrix object for storing co-occurrence counts and marginal counts
    print('creating datamatrix object for storing co-occurrence counts and marginal counts...')
    term_term = dataclasses.datamatrix(rowname='term_dictidname',
                                       rowlabels=rowlabels.copy(),
                                       rowmeta=copy.deepcopy(rowmeta),
                                       columnname='term_dictidname',
                                       columnlabels=columnlabels.copy(),
                                       columnmeta=copy.deepcopy(columnmeta),
                                       matrixname='literature_cooccurrence_from_termite',
                                       matrix=np.zeros((rowlabels.size, columnlabels.size), dtype='int64'))
    del rowlabels, rowmeta, columnlabels, columnmeta
    print(term_term)
    
    # get co-occurrence counts and marginal counts
    print('calculating co-occurrence counts and marginal counts...')
    row_pmids_intersectionunion = defaultdict(set) # the set of PMIDs mentioning row term i and any column term (union of all of the intersections)
    column_pmids_intersectionunion = defaultdict(set) # the set of PMIDs mentioning column term j and any row term (union of all of the intersections)
    all_pmids_intersectionunion = set() # the set of PMIDs mentioning any row term AND any column term ("universe" is limited to publications that have at least one row term association AND at least one column term association)
    all_pmids_union = set() # the set of PMIDs mentioning any row term OR any column term ("universe" is limited to publications that have at least one row term association OR at least one column term association)
    # *** term_term_union_matrix = np.zeros(term_term.shape, dtype='int64') # the count of PMIDs mentioning row term i OR column term j
    for i, rowlabel in enumerate(term_term.rowlabels):
        if np.mod(i, 100) == 0 or i+1 == term_term.shape[0]:
            print('working on row {0!s} of {1!s}...'.format(i+1, term_term.shape[0]), flush=True)
        row_pmids = rowterm_pmids[rowlabel]
        for j, columnlabel in enumerate(term_term.columnlabels):
            column_pmids = columnterm_pmids[columnlabel]
            intersection_pmids = row_pmids.intersection(column_pmids)
            term_term.matrix[i,j] = len(intersection_pmids) # the count of PMIDs mentioning row term i AND column term j
    #        all_pmids_union = row_pmids.union(column_pmids)
    #        term_term_union_matrix[i,j] = len(all_pmids_union) # the count of PMIDs mentioning row term i OR column term j
            if rowlabel != columnlabel:
                row_pmids_intersectionunion[rowlabel].update(intersection_pmids)
                column_pmids_intersectionunion[columnlabel].update(intersection_pmids)
        all_pmids_union.update(row_pmids)
        all_pmids_intersectionunion.update(row_pmids_intersectionunion[rowlabel])
    for column_pmids in columnterm_pmids.values():
        all_pmids_union.update(column_pmids)
    
    # include marginal counts as metadata
    print('including marginal counts as datamatrix metadata...')
    #     relevant universe
    term_term.rowmeta['term_count_intersectionunion'] = np.array([len(row_pmids_intersectionunion[rowlabel]) for rowlabel in term_term.rowlabels], dtype='int64')
    term_term.columnmeta['term_count_intersectionunion'] = np.array([len(column_pmids_intersectionunion[columnlabel]) for columnlabel in term_term.columnlabels], dtype='int64')
    term_term.rowmeta['all_count_intersectionunion'] = np.full(term_term.shape[0], len(all_pmids_intersectionunion), dtype='int64')
    term_term.columnmeta['all_count_intersectionunion'] = np.full(term_term.shape[1], len(all_pmids_intersectionunion), dtype='int64')
    #     whole universe
    term_term.rowmeta['term_count_union'] = np.array([len(rowterm_pmids[rowlabel]) for rowlabel in term_term.rowlabels], dtype='int64')
    term_term.columnmeta['term_count_union'] = np.array([len(columnterm_pmids[columnlabel]) for columnlabel in term_term.columnlabels], dtype='int64')
    term_term.rowmeta['all_count_union'] = np.full(term_term.shape[0], len(all_pmids_union), dtype='int64')
    term_term.columnmeta['all_count_union'] = np.full(term_term.shape[1], len(all_pmids_union), dtype='int64')
    
    # *** no need to calculate term_term_union_matrix
    #     if want this as universe size,
    #        start with universe size = all_count_intersectionunion
    #        calculate true positive, true negatives, false positives, false negatives
    #        subtract true negatives from universe size and set true negatives to zero
    
    # save results
    print('saving results...')
    datasetIO.save_datamatrix('{0}_{1}_datamatrix_pmidcounts_year_{2}_datestamp_{3}_minscore_{4!s}.txt.gz'.format(row_dictionary, column_dictionary, year, datestamp, min_score), term_term)
    datasetIO.save_datamatrix('{0}_{1}_datamatrix_pmidcounts_year_{2}_datestamp_{3}_minscore_{4!s}.pickle'.format(row_dictionary, column_dictionary, year, datestamp, min_score), term_term)

    print('done count_term-term_pmids_from_termite.py', flush=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get term-term pmid counts and marginal counts from termite.')
    parser.add_argument('--dictionaries', help='two dictionaries of biomedical terms, first for row terms and second for column terms', type=str, nargs='+')
    parser.add_argument('--year', help='year of termite file', type=str, default='all')
    parser.add_argument('--datestamp', help='datestamp of termite file', type=str, default='all')
    parser.add_argument('--min_score', help='min score for filtering term-pmid associations', type=int, default=2)
    args = parser.parse_args()
    main(args.dictionaries, args.year, args.datestamp, args.min_score)