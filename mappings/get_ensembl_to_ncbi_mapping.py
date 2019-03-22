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

import gzip
import pickle
from collections import defaultdict

ncbigeneid_ncbigenesym = defaultdict(set)
ncbigenesym_ncbigeneid = defaultdict(set)
with gzip.open('Homo_sapiens.gene_info.gz', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    for line in fr:
        ncbi_tax_id, ncbi_gene_id, ncbi_gene_sym = [x.strip() for x in line.split('\t', maxsplit=3)[:3]]
        if ncbi_tax_id == '9606':
            ncbi_gene_sym = ncbi_gene_sym.upper()
            ncbigeneid_ncbigenesym[ncbi_gene_id].add(ncbi_gene_sym)
            ncbigenesym_ncbigeneid[ncbi_gene_sym].add(ncbi_gene_id)
        
num_nonunique_id2sym_mappings = sum([len(v) > 1 for v in ncbigeneid_ncbigenesym.values()])
assert num_nonunique_id2sym_mappings == 0

num_nonunique_sym2id_mappings = sum([len(v) > 1 for v in ncbigenesym_ncbigeneid.values()])

discard_ncbi_gene_ids = set()
discard_ncbi_gene_syms = set()
with open('Homo_sapiens.gene_info.non_unique.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    fw.write('\t'.join(['ncbi_gene_id', 'ncbi_gene_sym']) + '\n')
    for ncbi_gene_sym, ncbi_gene_ids in ncbigenesym_ncbigeneid.items():
        ncbi_gene_ids = list(ncbi_gene_ids)
        if len(ncbi_gene_ids) > 1:
            discard_ncbi_gene_syms.add(ncbi_gene_sym)
            discard_ncbi_gene_ids.update(ncbi_gene_ids)
            for ncbi_gene_id in ncbi_gene_ids:
                fw.write('\t'.join([ncbi_gene_id, ncbi_gene_sym]) + '\n')
                
for ncbi_gene_id in discard_ncbi_gene_ids:
    del ncbigeneid_ncbigenesym[ncbi_gene_id]
for ncbi_gene_sym in discard_ncbi_gene_syms:
    del ncbigenesym_ncbigeneid[ncbi_gene_sym]

ncbigeneid_ncbigenesym = dict(ncbigeneid_ncbigenesym)
for k, v in ncbigeneid_ncbigenesym.items():
    ncbigeneid_ncbigenesym[k] = v.pop()
ncbigenesym_ncbigeneid = dict(ncbigenesym_ncbigeneid)
for k, v in ncbigenesym_ncbigeneid.items():
    ncbigenesym_ncbigeneid[k] = v.pop()

with open('ncbigeneid_ncbigenesym_dict.pickle', 'wb') as fw:
    pickle.dump(ncbigeneid_ncbigenesym, fw)
with open('ncbigenesym_ncbigeneid_dict.pickle', 'wb') as fw:
    pickle.dump(ncbigenesym_ncbigeneid, fw)

ensembl_ncbi = defaultdict(set)
with open('ensembl_to_ncbi.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    for line in fr:
        ensembl_gene_id, hgnc_gene_sym, ensembl_gene_name, ncbi_gene_id = [x.strip() for x in line.split('\t')]
        if len(hgnc_gene_sym) > 0:
            gene_sym = hgnc_gene_sym.upper()
        else:
            gene_sym = ensembl_gene_name.upper()
        if ncbi_gene_id in ncbigeneid_ncbigenesym:
            ncbi_gene_sym = ncbigeneid_ncbigenesym[ncbi_gene_id]
        else:
            ncbi_gene_sym = 'NOTFOUND'
        ensembl_ncbi[ensembl_gene_id].add((gene_sym, ncbi_gene_id, ncbi_gene_sym))
        
num_nonunique_mappings = sum([len(v) > 1 for v in ensembl_ncbi.values()])

discard_ensembl_gene_ids = set()
with open('ensembl_to_ncbi_non_unique.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw1,\
     open('ensembl_to_ncbi_no_ncbigenesym.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw2:
    fw1.write('\t'.join(['ensembl_gene_id', 'gene_sym', 'ncbi_gene_id', 'ncbi_gene_sym']) + '\n')
    fw2.write('\t'.join(['ensembl_gene_id', 'gene_sym', 'ncbi_gene_id', 'ncbi_gene_sym']) + '\n')
    for ensembl_gene_id, id_triples in ensembl_ncbi.items():
        id_triples = list(id_triples)
        if len(id_triples) > 1:
            keep_id_triples = set()
            for gene_sym, ncbi_gene_id, ncbi_gene_sym in id_triples:
                fw1.write('\t'.join([ensembl_gene_id, gene_sym, ncbi_gene_id, ncbi_gene_sym]) + '\n')
                if gene_sym == ncbi_gene_sym:
                    keep_id_triples.add((gene_sym, ncbi_gene_id, ncbi_gene_sym))
            keep_id_triples = list(keep_id_triples)
            if len(keep_id_triples) == 1:
                ensembl_ncbi[ensembl_gene_id] = keep_id_triples[0][1:]
            else:
                discard_ensembl_gene_ids.add(ensembl_gene_id)
        elif id_triples[0][2] == 'NOTFOUND':
            fw2.write('\t'.join([ensembl_gene_id] + list(id_triples[0])) + '\n')
            discard_ensembl_gene_ids.add(ensembl_gene_id)
        else:
            ensembl_ncbi[ensembl_gene_id] = id_triples[0][1:]
        
num_discarded_ensembl_gene_ids = len(discard_ensembl_gene_ids)

ensembl_ncbi = dict(ensembl_ncbi)
for ensembl_gene_id in discard_ensembl_gene_ids:
    del ensembl_ncbi[ensembl_gene_id]

with open('ensemblgeneid_ncbigeneidsym_dict.pickle', 'wb') as fw:
    pickle.dump(ensembl_ncbi, fw)


