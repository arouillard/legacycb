# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biology
GSK
"""

import pickle
from collections import defaultdict

print('parsing ontology...', flush=True)
hit = False
mpid_name = {}
child_parent = defaultdict(set)
with open('../../original_data/impc/MPheno_OBO.ontology', 'rt') as fr:
    for line in fr:
        if '[Term]' in line:
            hit = True
            field_value = {}
        elif hit and line == '\n':
            hit = False
            parents = set()
            for field, value in field_value.items():
                if field == 'id':
                    mp_id = value
                    child = value
                elif field == 'name':
                    name = value
                elif field == 'is_a':
                    parents.add(value)
            mpid_name[mp_id] = name
            if len(parents) > 0:
                child_parent[child].update(parents)
        elif hit:
            field, value = [x.strip() for x in line.split(': ')[:2]]
            if ' ! ' in value:
                value = value.split(' ! ')[0].strip()
            field_value[field] = value

print('saving term ids and names...', flush=True)
with open('../../original_data/impc/mpid_name_dict.pickle', 'wb') as fw:
    pickle.dump(mpid_name, fw)

print('saving child parent mappings', flush=True)
with open('../../original_data/impc/child_parent_mouse-phenotype-ids_dict.pickle', 'wb') as fw:
    pickle.dump(child_parent, fw)

print('done.', flush=True)
