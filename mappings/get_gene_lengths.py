# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import sys
sys.path.append('../utilities')

import pickle
import numpy as np
from collections import defaultdict

ensembl_length = defaultdict(list)
with open('gencode.v19.genes.v7.patched_contigs.gtf', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    for i in range(6):
        fr.readline()
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        if entries[2] == 'gene':
            length = float(entries[4]) - float(entries[3])
            metadata = [x.strip() for x in entries[8].split('; ')]
            ensembl_gene_id = metadata[0].replace('gene_id "', '').replace('"', '')
            ensembl_gene_name = metadata[4].replace('gene_name "', '').replace('"', '')
            ensembl_length[(ensembl_gene_id, ensembl_gene_name)].append(length)
lengthspergene = np.array([len(v) for v in ensembl_length.values()], dtype='int64')
assert (lengthspergene == 1).all()
for k, v in ensembl_length.items():
    ensembl_length[k] = v[0]
with open('ensembl_geneidname_length_dict.pickle', mode='wb') as fw:
    pickle.dump(ensembl_length, fw)
