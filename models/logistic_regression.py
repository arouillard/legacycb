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

import json
import copy
import numpy as np
import datasetIO
from dataclasses import datamatrix as DataMatrix
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, log_loss
from sklearn.model_selection import LeaveOneOut


def main(datamatrix_path, test_index, response_variable_name, valid_index, valid_fraction, feature_fraction, regularization_type, inverse_regularization_strength, intercept_scaling, pos_neg_weight_ratio, evaluation_statistic, save_weights, save_folder, datamatrix):

    print('loading datamatrix...', flush=False)
    if datamatrix == None or type(datamatrix) == str:
        dm = datasetIO.load_datamatrix(datamatrix_path)
    else:
        dm = datamatrix

    print('setting random seed with test_index {0!s}...'.format(test_index), flush=False)
    np.random.seed(test_index)

    print('getting bootstrap sample...', flush=False)
    all_indices = np.arange(dm.shape[0])
    boot_indices = np.random.choice(dm.shape[0], dm.shape[0], replace=True)
    test_indices = all_indices[~np.in1d(all_indices, boot_indices)]

    print('reserving out-of-bag samples as test set...', flush=False)
    Y = {'test':dm.rowmeta[response_variable_name][test_indices].astype('bool')}
    X = {'test':dm.matrix[test_indices,:]}
    
    print('setting random seed with valid_index {0!s}...'.format(valid_index), flush=False)
    np.random.seed(valid_index)

    print('splitting bootstrap sample into training and validation sets...', flush=False)
    if type(valid_fraction) == str and (valid_fraction.lower() == 'loo' or valid_fraction.lower() == 'loocv'):
        valid_fraction = 'loo'
        valid_indices = all_indices
        train_indices = all_indices
    else:
        valid_indices = np.random.choice(dm.shape[0], round(valid_fraction*dm.shape[0]), replace=False)
        train_indices = all_indices[~np.in1d(all_indices, valid_indices)]

    Y['train'] = dm.rowmeta[response_variable_name][boot_indices][train_indices].astype('bool')
    Y['valid'] = dm.rowmeta[response_variable_name][boot_indices][valid_indices].astype('bool')
    X['train'] = dm.matrix[boot_indices,:][train_indices,:]
    X['valid'] = dm.matrix[boot_indices,:][valid_indices,:]
    
    print('fitting and evaluating models...', flush=False)
    stages = ['validation', 'testing']
    data_subsets = ['fit', 'predict']
    performance_stats = ['auroc', 'auprc', 'brier', 'nll', 'tp', 'fn', 'tn', 'fp', 'ap', 'an', 'pp', 'pn', 'n', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'fdr', 'npv', 'fomr', 'acc', 'mcr', 'prev', 'plr', 'nlr', 'dor', 'drr', 'darr', 'mrr', 'marr', 'mcc', 'fnlp', 'f1', 'f1_100', 'f1_50', 'f1_25', 'f1_10', 'f1_5', 'f1_3', 'f1_2', 'f2', 'f3', 'f5', 'f10', 'f25', 'f50', 'f100']
    if valid_fraction == 'loo':
        X.update({'validation':{'fit':X['train'], 'predict':X['valid']}, 'testing':{'fit':X['train'], 'predict':X['test']}})
        Y.update({'validation':{'fit':Y['train'], 'predict':Y['valid']}, 'testing':{'fit':Y['train'], 'predict':Y['test']}})
    else:
        X.update({'validation':{'fit':X['train'], 'predict':X['valid']}, 'testing':{'fit':np.append(X['train'], X['valid'], 0), 'predict':X['test']}})
        Y.update({'validation':{'fit':Y['train'], 'predict':Y['valid']}, 'testing':{'fit':np.append(Y['train'], Y['valid']), 'predict':Y['test']}})
    stat_subset = {}
    for stage in stages:
        print('working on {0} stage...'.format(stage), flush=False)
        
        if feature_fraction < 1:
            print('performing univariate feature selection...', flush=False)
            num_features = round(feature_fraction*dm.shape[1])
            test_stats, p_values = ttest_ind(X[stage]['fit'][Y[stage]['fit'],:], X[stage]['fit'][~Y[stage]['fit'],:], axis=0, equal_var=False, nan_policy='propagate')
            ranks = np.argsort(p_values)
            selected_indices = ranks[:num_features]
            selected_features = dm.columnlabels[selected_indices]
            if stage == 'testing':
                print('plotting univariate test statistics...', flush=False)
                plt.figure()
                plt.hist(test_stats, 50)
                plt.savefig('{0}/univariate_test_statistics.png'.format(save_folder), transparent=True, pad_inches=0, dpi=100)
                plt.figure()
                plt.hist(p_values, 50)
                plt.savefig('{0}/univariate_pvalues.png'.format(save_folder), transparent=True, pad_inches=0, dpi=100)
                plt.figure()
                plt.hist(-np.log10(p_values), 50)
                plt.savefig('{0}/univariate_nlps.png'.format(save_folder), transparent=True, pad_inches=0, dpi=100)
        else:
            print('skipping univariate feature selection...', flush=False)
            selected_indices = np.arange(dm.shape[1], dtype='int64')
            selected_features = dm.columnlabels.copy()
        print('selected {0!s} features...'.format(selected_features.size), flush=False)
        
        print('calculating class weights...', flush=False)
        pos_weight = np.sqrt(pos_neg_weight_ratio)*((Y[stage]['fit'].size)/2/(Y[stage]['fit'].sum())) # (assign weight to class)*(adjust for unbalanced classes)
        neg_weight = (1/pos_weight)*((Y[stage]['fit'].size)/2/((~Y[stage]['fit']).sum())) # (assign weight to class)*(adjust for unbalanced classes)
        class_weight = {True:pos_weight, False:neg_weight}
        
        print('fitting model...', flush=False)
        logistic_regression_model = LogisticRegression(penalty=regularization_type, C=inverse_regularization_strength, intercept_scaling=intercept_scaling, class_weight=class_weight).fit(X[stage]['fit'][:,selected_indices], Y[stage]['fit'])
        
        if stage == 'testing':
            print('plotting feature weights...', flush=False)
            iter_feature = DataMatrix(rowname='iteration',
                                      rowlabels=np.array(['test{0!s}_valid{1!s}'.format(test_index, valid_index)], dtype='object'),
                                      rowmeta={'intercept':logistic_regression_model.intercept_, 'test_index':np.array([test_index], dtype='int64'), 'valid_index':np.array([valid_index], dtype='int64')},
                                      columnname=dm.columnname,
                                      columnlabels = dm.columnlabels.copy(),
                                      columnmeta=copy.deepcopy(dm.columnmeta),
                                      matrixname='feature_weights',
                                      matrix=np.zeros((1, dm.shape[1]), dtype='float64'))
            feature_idx = {f:i for i,f in enumerate(dm.columnlabels)}
            for feature, weight in zip(selected_features, logistic_regression_model.coef_[0,:]):
                iter_feature.matrix[0,feature_idx[feature]] = weight
            plt.figure()
            plt.hist(iter_feature.matrix[0,:], 50)
            plt.savefig('{0}/feature_weights.png'.format(save_folder), transparent=True, pad_inches=0, dpi=100)
            if feature_fraction < 1:
                plt.figure()
                plt.hist(iter_feature.matrix[0,selected_indices], 50)
                plt.savefig('{0}/feature_weights_selected.png'.format(save_folder), transparent=True, pad_inches=0, dpi=100)
            
            if save_weights:
                print('saving feature weights...', flush=False)
                datasetIO.save_datamatrix('{0}/iter_feature_datamatrix.txt.gz'.format(save_folder), iter_feature)

        print('creating datamatrix for performance statistics...', flush=False)
        stat_subset[stage] = DataMatrix(rowname='performance_statistic',
                                    rowlabels=np.array(performance_stats, dtype='object'),
                                    rowmeta={},
                                    columnname='data_subset',
                                    columnlabels = np.array(data_subsets, dtype='object'),
                                    columnmeta={},
                                    matrixname='classifier_performance_on_data_subsets',
                                    matrix=np.zeros((len(performance_stats), len(data_subsets)), dtype='float64'))

        for j, subset in enumerate(stat_subset[stage].columnlabels):
            print('evaluating performance on {0} subset...'.format(subset), flush=False)
            if valid_fraction == 'loo' and stage == 'validation' and subset == 'predict':
                P_pred = np.zeros(X[stage][subset].shape[0], dtype='float64')
                for train_index, test_index in LeaveOneOut().split(X[stage][subset]):
                    logistic_regression_model = LogisticRegression(penalty=regularization_type, C=inverse_regularization_strength, intercept_scaling=intercept_scaling, class_weight=class_weight).fit(X[stage]['fit'][train_index,:][:,selected_indices], Y[stage]['fit'][train_index])
                    P_pred[test_index] = logistic_regression_model.predict_proba(X[stage][subset][test_index,:][:,selected_indices])[:,logistic_regression_model.classes_==1][0][0]
            else:
                P_pred = logistic_regression_model.predict_proba(X[stage][subset][:,selected_indices])[:,logistic_regression_model.classes_==1]
            Y_pred = P_pred > 0.5
            
            auroc = roc_auc_score(Y[stage][subset], P_pred)
            auprc = average_precision_score(Y[stage][subset], P_pred)
            brier = brier_score_loss(Y[stage][subset], P_pred)
            nll = log_loss(Y[stage][subset], P_pred)
            
            tn, fp, fn, tp = confusion_matrix(Y[stage][subset], Y_pred).ravel()
            
            # incorporate a prior with effective sample size = n_eff, where prior represents random predictions
            n_eff = 1
            prevalence = (tp + fn)/(tn + fp + fn + tp)
            tp += n_eff*prevalence/2
            fn += n_eff*prevalence/2
            tn += n_eff*(1-prevalence)/2
            fp += n_eff*(1-prevalence)/2
            
            ap = tp + fn
            an = fp + tn
            pp = tp + fp
            pn = tn + fn
            n = tn + fp + fn + tp
            
            tpr = tp/ap # sensitivity, recall
            fnr = fn/ap # 1-tpr, 1-sensitivity, 1-recall
            tnr = tn/an # specificity
            fpr = fp/an # 1-tnr, 1-specificity
            
            ppv = tp/pp # precision
            fdr = fp/pp # 1-ppv, 1-precision
            npv = tn/pn
            fomr = fn/pn # 1-npv
            
            acc = (tp + tn)/n
            mcr = (fp + fn)/n # 1-acc
            prev = ap/n
            
            plr = (tp/fp)/(ap/an) # tpr/fpr, sensitivity/(1-specificity), ratio of positives to negatives in positive predictions relative to ratio in whole sample, higher is better
            nlr = (fn/tn)/(ap/an) # fnr/tnr, (1-sensitivity)/specificity, ratio of positives to negatives in negative predictions relative to ratio in whole sample, lower is better
            dor = (tp/fp)/(fn/tn) # plr/nlr, ratio of positives to negatives in positive predictions, divided by ratio of positives to negatives in negative predictions
            drr = (tp/pp)/(fn/pn) # ppv/fomr, relative risk or risk ratio, fraction of positives in positive predictions divided by fraction of positives in negative predictions
            darr = (tp/pp) - (fn/pn) # ppv - fomr, absolute risk reduction, fraction of positives in positive predictions minus fraction of positives in negative predictions
            mrr = (tp/pp)/(ap/n) # ppv/prev, modified (by me) relative risk or risk ratio, fraction of positives in positive predictions divided by fraction of positives in whole sample
            marr = (tp/pp) - (ap/n) # ppv - prev, modified (by me) absolute risk reduction, fraction of positives in positive predictions minus fraction of positives in whole sample
            
            mcc = (tp*tn - fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            fnlp = -hypergeom.logsf(tp, n, ap, pp, loc=1)/np.log(10)
            
            precision = ppv
            recall = tpr
            f1 = (1 + (1**2))*precision*recall/((1**2)*precision + recall)
            f1_100 = (1 + (1/100**2))*precision*recall/((1/100**2)*precision + recall)
            f1_50 = (1 + (1/50**2))*precision*recall/((1/50**2)*precision + recall)
            f1_25 = (1 + (1/25**2))*precision*recall/((1/25**2)*precision + recall)
            f1_10 = (1 + (1/10**2))*precision*recall/((1/10**2)*precision + recall)
            f1_5 = (1 + (1/5**2))*precision*recall/((1/5**2)*precision + recall)
            f1_3 = (1 + (1/3**2))*precision*recall/((1/3**2)*precision + recall)
            f1_2 = (1 + (1/2**2))*precision*recall/((1/2**2)*precision + recall)
            f2 = (1 + (2**2))*precision*recall/((2**2)*precision + recall)
            f3 = (1 + (3**2))*precision*recall/((3**2)*precision + recall)
            f5 = (1 + (5**2))*precision*recall/((5**2)*precision + recall)
            f10 = (1 + (10**2))*precision*recall/((10**2)*precision + recall)
            f25 = (1 + (25**2))*precision*recall/((25**2)*precision + recall)
            f50 = (1 + (50**2))*precision*recall/((50**2)*precision + recall)
            f100 = (1 + (100**2))*precision*recall/((100**2)*precision + recall)
            
            stat_subset[stage].matrix[:,j] = [auroc, auprc, brier, nll, tp, fn, tn, fp, ap, an, pp, pn, n, tpr, fnr, tnr, fpr, ppv, fdr, npv, fomr, acc, mcr, prev, plr, nlr, dor, drr, darr, mrr, marr, mcc, fnlp, f1, f1_100, f1_50, f1_25, f1_10, f1_5, f1_3, f1_2, f2, f3, f5, f10, f25, f50, f100]
        
        print('saving performance statistics...', flush=False)
        datasetIO.save_datamatrix('{0}/stat_subset_datamatrix_{1}.txt.gz'.format(save_folder, stage), stat_subset[stage])
        
        print('printing performance statistics...', flush=False)
        print('\t'.join(['stage', stat_subset[stage].rowname] + stat_subset[stage].columnlabels.tolist()), flush=False)
        for stat, vals in zip(stat_subset[stage].rowlabels, stat_subset[stage].matrix):
            print('\t'.join([stage, stat] + ['{0:1.3g}'.format(v) for v in vals]), flush=False)
    
    print('saving evaluation statistic...', flush=False)
    objective = stat_subset['validation'].select(evaluation_statistic, 'predict')
    with open('{0}/output.json'.format(save_folder), mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        json.dump(objective, fw, indent=2)
    
    print('done logistic_regression.py', flush=False)

