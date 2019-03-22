# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import os
import argparse
import pickle
import gzip
import json
import copy
import numpy as np
import tsdae_apply_functions
import modelevaluation
import datasetIO
import matplotlib.pyplot as plt


def main(reconstructions_path):
    
    # read reconstructions
    print('reading reconstructions...', flush=True)
    designpath_selectedstep = []
    with open(reconstructions_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        for line in fr:
            designpath_selectedstep.append([x.strip() for x in line.split('\t')])
    print('found {0!s} reconstructions...'.format(len(designpath_selectedstep)), flush=True)
    
    # evaluate reconstructions
    print('evaluating reconstructions...', flush=True)
    for didx, (design_path, selected_step) in enumerate(designpath_selectedstep):
        print('working on {0}...'.format(design_path), flush=True)
        print('selected step:{0!s}...'.format(selected_step), flush=True)
        
        
        # load design
        print('loading design...', flush=True)
        with open(design_path, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
            d = json.load(fr)
        if 'apply_activation_to_embedding' not in d: # for legacy code
            d['apply_activation_to_embedding'] = True
        if 'use_batchnorm' not in d: # for legacy code
            d['use_batchnorm'] = False
        if 'skip_layerwise_training' not in d: # for legacy code
            d['skip_layerwise_training'] = False
        phase = d['training_schedule'][-1]
        d['current_hidden_layer'] = phase['hidden_layer']
        d['current_finetuning_run'] = phase['finetuning_run']
        d['current_epochs'] = phase['epochs']
        
        
        # load data
        if didx == 0:
            print('loading data...', flush=True)
            partitions = ['train', 'valid', 'test']
            dataset = {}
            for partition in partitions:
                dataset[partition] = datasetIO.load_datamatrix('{0}/{1}.pickle'.format(d['input_path'], partition))
#                if 'all' not in dataset:
#                    dataset['all'] = copy.deepcopy(dataset[partition])
#                else:
#                    dataset['all'].append(dataset[partition], 0)

            # get parameters for marginal distributions
            # will sample from marginal distributions to impute missing values
            # for binary features, model as bernoulli (columnmeta['likelihood'] == 'bernoulli')
            # for other features, model as gaussian
            marginalprobabilities = (1 + np.nansum(dataset['train'].matrix, 0, keepdims=True))/(2 + np.sum(~np.isnan(dataset['train'].matrix), 0, keepdims=True)) # posterior mean of beta-bernoulli with prior a=b=1
            marginalstdvs = np.nanstd(dataset['train'].matrix, 0, keepdims=True)
            isbernoullimarginal = (dataset['train'].columnmeta['likelihood'] == 'bernoulli').astype('float64').reshape(1,-1)
        
        
        # load reconstructions and evaluate performance
        recon = {}
        stat_cut = {}
        for partition in ['valid', 'test']: # partitions:
            print('working on partition {0}...'.format(partition), flush=True)
            print('loading reconstructions...', flush=True)
            recon[partition] = datasetIO.load_datamatrix('{0}/{1}_intermediate_reconstructions_layer{2!s}_finetuning{3!s}_step{4!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step))
            hitmat = np.logical_and(isbernoullimarginal.astype('bool'), np.isfinite(dataset[partition].matrix))
            print('evaluating performance...', flush=True)
#            stat_cut[partition] = modelevaluation.get_classifier_performance_stats(dataset[partition].matrix[hitmat].astype('bool'), recon[partition].matrix[hitmat], uP=1000, classifier_stats='all', plot_curves=False, get_priority_cutoffs=True, pp_min_frac=0.1, xx_min_frac=0.01)
#            print('saving performance...', flush=True)
#            datasetIO.save_datamatrix('{0}/{1}_intermediate_performance_layer{2!s}_finetuning{3!s}_step{4!s}.pickle'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), stat_cut[partition])
#            datasetIO.save_datamatrix('{0}/{1}_intermediate_performance_layer{2!s}_finetuning{3!s}_step{4!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), stat_cut[partition])
            with gzip.open('{0}/{1}_intermediate_performance_per_row_layer{2!s}_finetuning{3!s}_step{4!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), 'wt') as fw:
                fw.write('\t'.join(['label', 'auroc', 'auprc', 'ap']) + '\n')
                for i, (hm, dm, rm, label) in enumerate(zip(hitmat, dataset[partition].matrix, recon[partition].matrix, dataset[partition].rowlabels)):
                    if hm.any():
                        sc = modelevaluation.get_classifier_performance_stats(dm[hm].astype('bool'), rm[hm], uP=5000, classifier_stats='all', plot_curves=False, get_priority_cutoffs=True, pp_min_frac=0.1, xx_min_frac=0.01)
                        fw.write('\t'.join([label] + ['{0:1.5g}'.format(x) for x in [sc.select('auroc',[])[0], sc.select('auprc',[])[0], sc.select('ap',[])[0]]]) + '\n')
            with gzip.open('{0}/{1}_intermediate_performance_per_col_layer{2!s}_finetuning{3!s}_step{4!s}.txt.gz'.format(d['output_path'], partition, d['current_hidden_layer'], d['current_finetuning_run'], selected_step), 'wt') as fw:
                fw.write('\t'.join(['label', 'auroc', 'auprc', 'ap']) + '\n')
                for i, (hm, dm, rm, label) in enumerate(zip(hitmat.T, dataset[partition].matrix.T, recon[partition].matrix.T, dataset[partition].columnlabels)):
                    if hm.any():
                        sc = modelevaluation.get_classifier_performance_stats(dm[hm].astype('bool'), rm[hm], uP=5000, classifier_stats='all', plot_curves=False, get_priority_cutoffs=True, pp_min_frac=0.1, xx_min_frac=0.01)
                        fw.write('\t'.join([label] + ['{0:1.5g}'.format(x) for x in [sc.select('auroc',[])[0], sc.select('auprc',[])[0], sc.select('ap',[])[0]]]) + '\n')
            
#            auroc = []
#            auprc = []
#            for i, (h, d, r) in enumerate(zip(hitmat.T, dataset[partition].matrix.T, recon[partition].matrix.T)):
#                if h.any():
#                    sc = modelevaluation.get_classifier_performance_stats(d[h].astype('bool'), r[h], uP=1000, classifier_stats='all', plot_curves=False, get_priority_cutoffs=True, pp_min_frac=0.1, xx_min_frac=0.01)
#                    auroc.append(sc.select('auroc',[])[0])
#                    auprc.append(sc.select('auprc',[])[0])
            
                    

    print('done evaluate_tsdae_reconstructions.', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize TSDAE features at selected steps for one or more models.')
    parser.add_argument('reconstructions_path', help='path to .txt file containing design paths and selected step for models to be visualized', type=str)
    args = parser.parse_args()
    main(args.reconstructions_path)

