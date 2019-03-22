# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import sys
sys.path.append('../../utilities')

import gzip
import numpy as np
import datasetIO
import dataclasses
from matplotlib import pyplot as plt

def alignment_indices(ref, qry):
    if ref.size != qry.size:
        raise ValueError('ref and qry have different sizes.')
    sorted_to_ref = np.argsort(np.argsort(ref))
    qry_to_sorted = np.argsort(qry)
    qry_to_ref = qry_to_sorted[sorted_to_ref]
    return qry_to_ref


subject_metadata = {}
subject_field_idx = {'subject_id':0, 'sex':1, 'age':2, 'hardy':3}
for field in subject_field_idx:
    subject_metadata[field] = np.empty(0, dtype='object')
with open('../../original_data/GTEXv6/GTEx_Data_V6_Annotations_SubjectPhenotypesDS.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        for field, idx in subject_field_idx.items():
            subject_metadata[field] = np.insert(subject_metadata[field], subject_metadata[field].size, entries[idx])

subject_metadata['sex'][subject_metadata['sex']=='1'] = 'male'
subject_metadata['sex'][subject_metadata['sex']=='2'] = 'female'
subject_metadata['hardy'][subject_metadata['hardy']=='0'] = 'ventilator case'
subject_metadata['hardy'][subject_metadata['hardy']=='1'] = 'violent and fast death'
subject_metadata['hardy'][subject_metadata['hardy']=='2'] = 'fast death of natural causes'
subject_metadata['hardy'][subject_metadata['hardy']=='3'] = 'intermediate death'
subject_metadata['hardy'][subject_metadata['hardy']=='4'] = 'slow death'
subject_metadata['coarse_age'] = subject_metadata['age'].copy()
subject_metadata['coarse_age'][subject_metadata['coarse_age'] == '20-29'] = '20-39'
subject_metadata['coarse_age'][subject_metadata['coarse_age'] == '30-39'] = '20-39'
subject_metadata['coarse_age'][subject_metadata['coarse_age'] == '40-49'] = '40-59'
subject_metadata['coarse_age'][subject_metadata['coarse_age'] == '50-59'] = '40-59'
subject_metadata['coarse_age'][subject_metadata['coarse_age'] == '60-69'] = '60-79'
subject_metadata['coarse_age'][subject_metadata['coarse_age'] == '70-79'] = '60-79'


sample_metadata = {}
sample_field_idx = {'sample_id':0, 'general_tissue':5, 'specific_tissue':6, 'analysis_freeze':17, 'autolysis_score':1, 'ischemic_time':8, 'mapping_rate':23, 'reads_mapped':34, 'intergenic_rate':35}
sample_field_dtype = {'sample_id':'object', 'general_tissue':'object', 'specific_tissue':'object', 'analysis_freeze':'object', 'autolysis_score':'float64', 'ischemic_time':'float64', 'mapping_rate':'float64', 'reads_mapped':'float64', 'intergenic_rate':'float64'}
for field, dtype in sample_field_dtype.items():
    sample_metadata[field] = np.empty(0, dtype=dtype)
with open('../../original_data/GTEXv6/GTEx_Data_V6_Annotations_SampleAttributesDS.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        for field, idx in sample_field_idx.items():
            if sample_field_dtype[field] == 'float64' and entries[idx] == '':
                entries[idx] = np.nan
            sample_metadata[field] = np.insert(sample_metadata[field], sample_metadata[field].size, entries[idx])

for field, dtype in sample_field_dtype.items():
    if dtype == 'float64':
        plt.figure()
        plt.hist(sample_metadata[field][~np.isnan(sample_metadata[field])])
        plt.xlabel(field)


for field in subject_metadata:
    sample_metadata[field] = np.empty(sample_metadata['sample_id'].size, dtype='object')
sample_metadata['subject_id'] = np.array(['-'.join(x.split('-', maxsplit=2)[:2]) for x in sample_metadata['sample_id']], dtype='object')
for i, subject_id in enumerate(subject_metadata['subject_id']):
    hit = sample_metadata['subject_id'] == subject_id
    for field in subject_metadata:
        sample_metadata[field][hit] = subject_metadata[field][i]
del subject_metadata


recount2_metadata = {}
recount2_field_idx = {'sample_id':19, 'auc':10, 'run_id':3}
recount2_field_dtype = {'sample_id':'object', 'auc':'float64', 'run_id':'object'}
for field, dtype in recount2_field_dtype.items():
    recount2_metadata[field] = np.empty(0, dtype=dtype)
with open('../../original_data/GTEXv6/SRP012682.tsv', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    for line in fr:
        entries = [x.strip() for x in line.split('\t')]
        for field, idx in recount2_field_idx.items():
            if recount2_field_dtype[field] == 'float64' and entries[idx] == '':
                entries[idx] = np.nan
            recount2_metadata[field] = np.insert(recount2_metadata[field], recount2_metadata[field].size, entries[idx])

for field, dtype in recount2_field_dtype.items():
    if dtype == 'float64':
        plt.figure()
        plt.hist(recount2_metadata[field][~np.isnan(recount2_metadata[field])])
        plt.xlabel(field)

recount2_metadata['subject_id'] = np.array(['-'.join(x.split('-', maxsplit=2)[:2]) for x in recount2_metadata['sample_id']], dtype='object')


with gzip.open('../../original_data/GTEXv6/counts_gene.tsv.gz', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    run_ids = np.array([x.strip() for x in fr.readline().split('\t')[:-1]], dtype='object')
    ensembl_gene_ids = np.empty(0, dtype='object')
    for line in fr:
        ensembl_gene_ids = np.insert(ensembl_gene_ids, ensembl_gene_ids.size, line.rsplit('\t', maxsplit=1)[1].strip())


si = alignment_indices(ref=run_ids, qry=recount2_metadata['run_id'])
for field, values in recount2_metadata.items():
    recount2_metadata[field] = values[si]

keep = np.in1d(sample_metadata['sample_id'], recount2_metadata['sample_id'])
for field, values in sample_metadata.items():
    sample_metadata[field] = values[keep]
si = alignment_indices(ref=recount2_metadata['sample_id'], qry=sample_metadata['sample_id'])
for field, values in sample_metadata.items():
    sample_metadata[field] = values[si]


assert (recount2_metadata['subject_id'] == sample_metadata['subject_id']).all()
assert (recount2_metadata['sample_id'] == sample_metadata['sample_id']).all()
assert (recount2_metadata['run_id'] == run_ids).all()


sample_metadata['run_id'] = recount2_metadata['run_id'].copy()
sample_metadata['auc'] = recount2_metadata['auc'].copy()
del recount2_metadata


tobediscarded = np.logical_or.reduce((sample_metadata['mapping_rate'] < 0.8,
                                      sample_metadata['ischemic_time'] < -12*60,
                                      sample_metadata['autolysis_score'] > 2.5,
                                      sample_metadata['intergenic_rate'] > 0.1,
                                      sample_metadata['auc'] > 3e10)) # 733

u_specific_tissues, c_specific_tissues = np.unique(sample_metadata['specific_tissue'][~tobediscarded], return_counts=True)
plt.figure(); plt.hist(c_specific_tissues, 50)
selected_specific_tissues = u_specific_tissues[c_specific_tissues > 30]

u_subject_ids, c_subject_ids = np.unique(sample_metadata['subject_id'][~tobediscarded], return_counts=True)
plt.figure(); plt.hist(c_subject_ids, 50)
selected_subjects = u_subject_ids[np.logical_and(c_subject_ids > 0.33*selected_specific_tissues.size, c_subject_ids < selected_specific_tissues.size)]

tobediscarded = np.logical_or.reduce((~np.in1d(sample_metadata['subject_id'], selected_subjects),
                                      ~np.in1d(sample_metadata['specific_tissue'], selected_specific_tissues),
                                      sample_metadata['mapping_rate'] < 0.8,
                                      sample_metadata['ischemic_time'] < -12*60,
                                      sample_metadata['autolysis_score'] > 2.5,
                                      sample_metadata['intergenic_rate'] > 0.1,
                                      sample_metadata['auc'] > 3e10)) # 4007

u_sample_specific_tissues = np.unique(sample_metadata['specific_tissue'][~tobediscarded])
u_subject_sexes = np.unique(sample_metadata['sex'][~tobediscarded])
u_subject_coarse_ages = np.unique(sample_metadata['coarse_age'][~tobediscarded])
'''
with open('../../original_data/GTEXv6/selected_sample_counts_by_specific_tissue.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as f1,\
     open('../../original_data/GTEXv6/selected_sample_counts_by_specific_tissue_and_sex.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as f2,\
     open('../../original_data/GTEXv6/selected_sample_counts_by_specific_tissue_and_sex_and_coarse_age.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as f3:
    for sample_specific_tissue in u_sample_specific_tissues:
        count = np.sum(np.logical_and(~tobediscarded, sample_metadata['specific_tissue'] == sample_specific_tissue))
        f1.write('\t'.join([sample_specific_tissue, str(count)]) + '\n')
        for subject_sex in u_subject_sexes:
            count = np.sum(np.logical_and(np.logical_and(~tobediscarded, sample_metadata['specific_tissue'] == sample_specific_tissue), sample_metadata['sex'] == subject_sex))
            f2.write('\t'.join([sample_specific_tissue, subject_sex, str(count)]) + '\n')
            for subject_coarse_age in u_subject_coarse_ages:
                count = np.sum(np.logical_and(np.logical_and(np.logical_and(~tobediscarded, sample_metadata['specific_tissue'] == sample_specific_tissue), sample_metadata['sex'] == subject_sex), sample_metadata['coarse_age'] == subject_coarse_age))
                f3.write('\t'.join([sample_specific_tissue, subject_sex, subject_coarse_age, str(count)]) + '\n')
'''

# semi-manually decide sampling scheme

'''
chosen_samples = np.empty(0, dtype='object')
with open('../../original_data/GTEXv6/sampling_scheme.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr,\
     open('../../original_data/GTEXv6/candidate_samples.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    fr.readline()
    for line in fr:
        sample_specific_tissue, subject_sex, subject_coarse_age, sample_total, sample_selected = [x.strip() for x in line.split('\t')]
        candidate_samples = sample_metadata['sample_id'][np.logical_and(np.logical_and(np.logical_and(~tobediscarded, sample_metadata['specific_tissue'] == sample_specific_tissue), sample_metadata['sex'] == subject_sex), sample_metadata['coarse_age'] == subject_coarse_age)]
        if candidate_samples.size != int(sample_total):
            raise ValueError('wrong count')
        if int(sample_selected) > 0:
            chosen_samples = np.append(chosen_samples, np.random.choice(candidate_samples, int(sample_selected), replace=False))
            fw.write('\n'.join(candidate_samples) + '\n')
with open('../../original_data/GTEXv6/chosen_samples.txt', mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
    fw.write('\n'.join(chosen_samples))
'''
chosen_samples = np.empty(0, dtype='object')
with open('../../original_data/GTEXv6/sampling_scheme.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    fr.readline()
    for line in fr:
        sample_specific_tissue, subject_sex, subject_coarse_age, sample_total, sample_selected = [x.strip() for x in line.split('\t')]
        candidate_samples = sample_metadata['sample_id'][np.logical_and(np.logical_and(np.logical_and(~tobediscarded, sample_metadata['specific_tissue'] == sample_specific_tissue), sample_metadata['sex'] == subject_sex), sample_metadata['coarse_age'] == subject_coarse_age)]
        if candidate_samples.size != int(sample_total):
            raise ValueError('wrong count')
        if int(sample_selected) > 0:
            chosen_samples = np.append(chosen_samples, np.random.choice(candidate_samples, int(sample_selected), replace=False))
with open('../../original_data/GTEXv6/chosen_samples.txt', mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
    chosen_samples = np.array(fr.read().split('\n'), dtype='object')

hit = np.in1d(sample_metadata['sample_id'], chosen_samples)
for field, values in sample_metadata.items():
    sample_metadata[field] = values[hit]
run_ids = run_ids[hit]

matrix = matrix = np.loadtxt('../../original_data/GTEXv6/counts_gene.tsv.gz', dtype='float64', delimiter='\t',
                             skiprows=1, usecols=hit.nonzero()[0], ndmin=2)

gene_tissue = dataclasses.datamatrix(rowname='ensembl_gene_id',
                                     rowlabels=ensembl_gene_ids,
                                     rowmeta={},
                                     columnname='recount2_run_id',
                                     columnlabels=run_ids,
                                     columnmeta=sample_metadata,
                                     matrixname='recount2_processed_rnaseq_counts_from_gtexv6',
                                     matrix=matrix)

datasetIO.save_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_counts.pickle', gene_tissue)
datasetIO.save_datamatrix('../../original_data/GTEXv6/gene_tissue_recount2gtexv6_chosen_samples_counts.txt.gz', gene_tissue)

