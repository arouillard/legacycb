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

import argparse
import numpy as np
import pandas as pd
#import dataclasses as dc
import datasetIO
import os
#import copy
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF as RBFKernel, Sum as SumKernel, Product as ProductKernel


def main(project_name, hyperparameters, evaluation_statistics, selection_criteria, sigma_multipliers):
    
    min_num_hp_combinations = 100
    num_gp_optimizer_restarts = 0 # 4
    outlier_sigma_multiplier = 6
    
    xline = np.linspace(0, 1, 100, dtype='float64')
    yline = np.linspace(0, 1, 100, dtype='float64')
    xmat, ymat = np.meshgrid(xline, yline)
    Xarr = np.append(xmat.reshape(-1,1), ymat.reshape(-1,1), 1)
    fxy = 2*Xarr[:,0]*Xarr[:,1]/(Xarr[:,0] + Xarr[:,1] + 1e-6)
    si = np.argsort(fxy)
    fxy = fxy[si]
    Xarr = Xarr[si,:]
    grid_indices = np.argsort(si)
    
    kernel = SumKernel(WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e3)), ProductKernel(ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e3)), RBFKernel(length_scale=np.array([1.0, 1.0], dtype='float64'), length_scale_bounds=(1e-2, 1e2))))
    
    project_folder = '../../hp_search/{0}'.format(project_name)
    print('project: {0}...'.format(project_name), flush=True)
    print('project_folder: {0}...'.format(project_folder), flush=True)
    
    search_folders = ['{0}/{1}'.format(project_folder, f) for f in os.listdir(project_folder) if f[:10] == 'hp_search_']
    search_ids = [int(f.rsplit('_', maxsplit=1)[-1]) for f in search_folders]
    print('found {0!s} search folders.'.format(len(search_folders)), flush=True)
        
    for search_id, search_folder in zip(search_ids, search_folders):
        print('working on search_folder: {0}...'.format(search_folder), flush=True)
        search_data_path = '{0}/hp_search_data.txt'.format(search_folder)
        search_data_path_with_stats = '{0}/hp_search_data_with_performance_stats.txt'.format(search_folder)
        print('search_data_path: {0}'.format(search_data_path), flush=True)
        if os.path.exists(search_data_path) and os.path.getsize(search_data_path) > 0:
            print('loading search data...', flush=True)
            df = pd.read_table(search_data_path, index_col=False)
            if df.shape[0] >= min_num_hp_combinations:
                print('appending performance stats...', flush=True)
                if os.path.exists(search_data_path_with_stats) and os.path.getsize(search_data_path) > 0:
                    df = pd.read_table(search_data_path_with_stats, index_col=False)
                else:
                    for stage in ['validation', 'testing']:
                        print('working on {0} stage...'.format(stage), flush=True)
                        for rowidx, combination_id in enumerate(df.combination_id):
                            combination_folder = '{0}/hp_combination_{1!s}'.format(search_folder, combination_id)
                            performance_data_path = '{0}/stat_subset_datamatrix_{1}.txt.gz'.format(combination_folder, stage)
                            if os.path.exists(performance_data_path):
                                stat_subset = datasetIO.load_datamatrix(performance_data_path)
                                if 'stat_mat' not in locals():
                                    stat_mat = np.full((df.shape[0], stat_subset.size), np.nan, dtype='float64')
                                    stat_cols = (stage + '_' + stat_subset.rowlabels.reshape(-1,1) + '_' + stat_subset.columnlabels.reshape(1,-1)).reshape(-1)
                                stat_mat[rowidx,:] = stat_subset.matrix.reshape(-1)
                        stat_df = pd.DataFrame(data=stat_mat, columns=stat_cols)
                        stat_df['combination_id'] = df.combination_id.values
                        df = df.set_index('combination_id').join(stat_df.set_index('combination_id')).reset_index()
                        del stat_mat, stat_cols, stat_df
                    df.to_csv(search_data_path_with_stats, sep='\t', index=False)
                
                if '{0}_search_domain'.format(hyperparameters[0]) not in df.columns:
                    df['{0}_search_domain'.format(hyperparameters[0])] = 0.5
                if '{0}_search_domain'.format(hyperparameters[1]) not in df.columns:
                    df['{0}_search_domain'.format(hyperparameters[1])] = 0.5
                if '{0}_model_space'.format(hyperparameters[0]) not in df.columns:
                    df['{0}_model_space'.format(hyperparameters[0])] = 1
                if '{0}_model_space'.format(hyperparameters[1]) not in df.columns:
                    df['{0}_model_space'.format(hyperparameters[1])] = 1
                
                for evaluation_statistic in evaluation_statistics:
                    print('working on performance evaluation statistic: {0}...'.format(evaluation_statistic), flush=True)
                    
                    C = df['combination_id'].values
                    Y_fit = df['validation_{0}_fit'.format(evaluation_statistic)].values
                    Y_fit = np.log10(Y_fit/(1-Y_fit))
                    Y_predict = df['validation_{0}_predict'.format(evaluation_statistic)].values
                    Y_predict = np.log10(Y_predict/(1-Y_predict))
                    Y_diff = Y_fit - Y_predict
                    X_1 = df['{0}_search_domain'.format(hyperparameters[0])].values
                    X_2 = df['{0}_search_domain'.format(hyperparameters[1])].values
                    keep = np.isfinite(np.concatenate((Y_fit.reshape(-1,1), Y_predict.reshape(-1,1), Y_diff.reshape(-1,1), X_1.reshape(-1,1), X_2.reshape(-1,1)), 1)).all(1)
                    C = C[keep]
                    Y_fit = Y_fit[keep]
                    Y_predict = Y_predict[keep]
                    Y_diff = Y_diff[keep]
                    X_1 = X_1[keep]
                    X_2 = X_2[keep]
                    X = np.append(X_1.reshape(-1,1), X_2.reshape(-1,1), 1)
                    
                    print('fitting Y_predict...', flush=True)
                    is_outlier = np.zeros(Y_predict.size, dtype='bool')
                    prev_outliers = -1
                    curr_outliers = 0
                    num_fits = 0
                    while curr_outliers - prev_outliers > 0 and not is_outlier.all():
                        gp_predict = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=num_gp_optimizer_restarts, normalize_y=True).fit(X[~is_outlier,:], Y_predict[~is_outlier])
                        Y_predict_hat_mean, Y_predict_hat_stdv = gp_predict.predict(X, return_std=True)
                        is_outlier = np.abs(Y_predict - Y_predict_hat_mean) > outlier_sigma_multiplier*Y_predict_hat_stdv
                        prev_outliers = curr_outliers
                        curr_outliers = is_outlier.sum()
                        num_fits += 1
                        print('num_fits', num_fits, 'curr_outliers', curr_outliers, 'prev_outliers', prev_outliers, flush=True)
                    Y_predict_hat_mean, Y_predict_hat_stdv = gp_predict.predict(Xarr, return_std=True)
                    plt.imsave('{0}/{1}_predict_hat_mean_4.png'.format(search_folder, evaluation_statistic), Y_predict_hat_mean[grid_indices].reshape(xmat.shape[0], xmat.shape[1]))
                    plt.imsave('{0}/{1}_predict_hat_stdv_4.png'.format(search_folder, evaluation_statistic), Y_predict_hat_stdv[grid_indices].reshape(xmat.shape[0], xmat.shape[1]))
                    
                    print('fitting Y_diff...', flush=True)
                    is_outlier = np.zeros(Y_diff.size, dtype='bool')
                    prev_outliers = -1
                    curr_outliers = 0
                    num_fits = 0
                    while curr_outliers - prev_outliers > 0 and not is_outlier.all():
                        gp_diff = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=num_gp_optimizer_restarts, normalize_y=True).fit(X[~is_outlier,:], Y_diff[~is_outlier])
                        Y_diff_hat_mean, Y_diff_hat_stdv = gp_diff.predict(X, return_std=True)
                        is_outlier = np.abs(Y_diff - Y_diff_hat_mean) > outlier_sigma_multiplier*Y_diff_hat_stdv
                        prev_outliers = curr_outliers
                        curr_outliers = is_outlier.sum()
                        num_fits += 1
                        print('num_fits', num_fits, 'curr_outliers', curr_outliers, 'prev_outliers', prev_outliers, flush=True)
                    Y_diff_hat_mean, Y_diff_hat_stdv = gp_diff.predict(Xarr, return_std=True)
                    plt.imsave('{0}/{1}_diff_hat_mean_4.png'.format(search_folder, evaluation_statistic), Y_diff_hat_mean[grid_indices].reshape(xmat.shape[0], xmat.shape[1]))
                    plt.imsave('{0}/{1}_diff_hat_stdv_4.png'.format(search_folder, evaluation_statistic), Y_diff_hat_stdv[grid_indices].reshape(xmat.shape[0], xmat.shape[1]))

                    for selection_criterion in selection_criteria:
                        print('working on selection criterion: {0}...'.format(selection_criterion), flush=True)
                        
                        for sigma_multiplier in sigma_multipliers:
                            print('working on sigma multiplier: {0}...'.format(sigma_multiplier), flush=True)
                            
                            if selection_criterion == 'optimistic_max':
                                # find hp combinations where Y_predict_hat_mean_max is within confidence interval of Y_predict_hat_mean
                                Y_predict_hat_mean_max = Y_predict_hat_mean.max()
                                Y_predict_hat_stdv_max = Y_predict_hat_stdv[Y_predict_hat_mean == Y_predict_hat_mean_max].mean()
                                hit = (Y_predict_hat_mean_max - Y_predict_hat_mean) <= (sigma_multiplier*Y_predict_hat_stdv + 1e-6)
                                # among these hits, find hp combinations where zero, or if no hits then Y_diff_hat_mean_min, is within confidence interval of Y_diff_hat_mean
                                Y_diff_hat_mean_min = np.min(np.abs(Y_diff_hat_mean[hit]))
                                Y_diff_hat_stdv_min = Y_diff_hat_stdv[hit][np.min(np.abs(Y_diff_hat_mean[hit])) == Y_diff_hat_mean_min].mean()
                                hit2 = np.logical_and(hit, np.abs(Y_diff_hat_mean) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6))
                                if not hit2.any():
                                    hit2 = np.logical_and(hit, (np.abs(Y_diff_hat_mean) - Y_diff_hat_mean_min) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6))
                                hit = hit2
                                # choose least regularized hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][-1]
                            elif selection_criterion == 'conservative_max':
                                # find hp combinations where Y_predict_hat_mean_max is within confidence interval of Y_predict_hat_mean
                                Y_predict_hat_mean_max = Y_predict_hat_mean.max()
                                Y_predict_hat_stdv_max = Y_predict_hat_stdv[Y_predict_hat_mean == Y_predict_hat_mean_max].mean()
                                hit = (Y_predict_hat_mean_max - Y_predict_hat_mean) <= (sigma_multiplier*Y_predict_hat_stdv + 1e-6)
                                # among these hits, find hp combinations where zero, or if no hits then Y_diff_hat_mean_min, is within confidence interval of Y_diff_hat_mean
                                Y_diff_hat_mean_min = np.min(np.abs(Y_diff_hat_mean[hit]))
                                Y_diff_hat_stdv_min = Y_diff_hat_stdv[hit][np.min(np.abs(Y_diff_hat_mean[hit])) == Y_diff_hat_mean_min].mean()
                                hit2 = np.logical_and(hit, np.abs(Y_diff_hat_mean) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6))
                                if not hit2.any():
                                    hit2 = np.logical_and(hit, (np.abs(Y_diff_hat_mean) - Y_diff_hat_mean_min) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6))
                                hit = hit2
                                # choose simplest hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][0]
                            elif selection_criterion == 'optimistic_match':
                                # find hp combinations where zero, or if no hits then Y_diff_hat_mean_min, is within confidence interval of Y_diff_hat_mean
                                Y_diff_hat_mean_min = np.min(np.abs(Y_diff_hat_mean))
                                Y_diff_hat_stdv_min = Y_diff_hat_stdv[np.min(np.abs(Y_diff_hat_mean)) == Y_diff_hat_mean_min].mean()
                                hit = np.abs(Y_diff_hat_mean) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6)
                                if not hit.any():
                                    hit = (np.abs(Y_diff_hat_mean) - Y_diff_hat_mean_min) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6)
                                # among these hits, find hp combinations where Y_predict_hat_mean_max is within confidence interval of Y_predict_hat_mean
                                Y_predict_hat_mean_max = Y_predict_hat_mean[hit].max()
                                Y_predict_hat_stdv_max = Y_predict_hat_stdv[hit][Y_predict_hat_mean[hit] == Y_predict_hat_mean_max].mean()
                                hit = np.logical_and(hit, (Y_predict_hat_mean_max - Y_predict_hat_mean) <= (sigma_multiplier*Y_predict_hat_stdv + 1e-6))
                                # choose least regularized hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][-1]
                            elif selection_criterion == 'conservative_match':
                                # find hp combinations where zero, or if no hits then Y_diff_hat_mean_min, is within confidence interval of Y_diff_hat_mean
                                Y_diff_hat_mean_min = np.min(np.abs(Y_diff_hat_mean))
                                Y_diff_hat_stdv_min = Y_diff_hat_stdv[np.min(np.abs(Y_diff_hat_mean)) == Y_diff_hat_mean_min].mean()
                                hit = np.abs(Y_diff_hat_mean) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6)
                                if not hit.any():
                                    hit = (np.abs(Y_diff_hat_mean) - Y_diff_hat_mean_min) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6)
                                # among these hits, find hp combinations where Y_predict_hat_mean_max is within confidence interval of Y_predict_hat_mean
                                Y_predict_hat_mean_max = Y_predict_hat_mean[hit].max()
                                Y_predict_hat_stdv_max = Y_predict_hat_stdv[hit][Y_predict_hat_mean[hit] == Y_predict_hat_mean_max].mean()
                                hit = np.logical_and(hit, (Y_predict_hat_mean_max - Y_predict_hat_mean) <= (sigma_multiplier*Y_predict_hat_stdv + 1e-6))
                                # choose simplest hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][0]
                            elif selection_criterion == 'optimistic_max_0':
                                # find hp combinations where Y_predict_hat_mean_max is within confidence interval of Y_predict_hat_mean
                                Y_predict_hat_mean_max = Y_predict_hat_mean.max()
                                Y_predict_hat_stdv_max = Y_predict_hat_stdv[Y_predict_hat_mean == Y_predict_hat_mean_max].mean()
                                hit = (Y_predict_hat_mean_max - Y_predict_hat_mean) <= (sigma_multiplier*Y_predict_hat_stdv + 1e-6)
                                # choose least regularized hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][-1]
                            elif selection_criterion == 'conservative_max_0':
                                # find hp combinations where Y_predict_hat_mean_max is within confidence interval of Y_predict_hat_mean
                                Y_predict_hat_mean_max = Y_predict_hat_mean.max()
                                Y_predict_hat_stdv_max = Y_predict_hat_stdv[Y_predict_hat_mean == Y_predict_hat_mean_max].mean()
                                hit = (Y_predict_hat_mean_max - Y_predict_hat_mean) <= (sigma_multiplier*Y_predict_hat_stdv + 1e-6)
                                # choose simplest hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][0]
                            elif selection_criterion == 'optimistic_match_0':
                                # find hp combinations where zero, or if no hits then Y_diff_hat_mean_min, is within confidence interval of Y_diff_hat_mean
                                Y_diff_hat_mean_min = np.min(np.abs(Y_diff_hat_mean))
                                Y_diff_hat_stdv_min = Y_diff_hat_stdv[np.min(np.abs(Y_diff_hat_mean)) == Y_diff_hat_mean_min].mean()
                                hit = np.abs(Y_diff_hat_mean) <= sigma_multiplier*Y_diff_hat_stdv + 1e-6
                                if not hit.any():
                                    hit = (np.abs(Y_diff_hat_mean) - Y_diff_hat_mean_min) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6)
                                # choose least regularized hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][-1]
                            elif selection_criterion == 'conservative_match_0':
                                # find hp combinations where zero, or if no hits then Y_diff_hat_mean_min, is within confidence interval of Y_diff_hat_mean
                                Y_diff_hat_mean_min = np.min(np.abs(Y_diff_hat_mean))
                                Y_diff_hat_stdv_min = Y_diff_hat_stdv[np.min(np.abs(Y_diff_hat_mean)) == Y_diff_hat_mean_min].mean()
                                hit = np.abs(Y_diff_hat_mean) <= sigma_multiplier*Y_diff_hat_stdv + 1e-6
                                if not hit.any():
                                    hit = (np.abs(Y_diff_hat_mean) - Y_diff_hat_mean_min) <= (sigma_multiplier*Y_diff_hat_stdv + 1e-6)
                                # choose simplest hp combination among the hits (***assumes lower index corresponds to simpler model***)
                                fxy_max = fxy[hit].max()
                                hit = np.logical_and(hit, (fxy_max - fxy) <= 1e-6)
                                hidx = hit.nonzero()[0][0]
                            else:
                                raise ValueError('invalid selection_criterion')
                            
                            X_1_hit, X_2_hit = Xarr[hidx,:]
                            d2 = (df['{0}_search_domain'.format(hyperparameters[0])].values - X_1_hit)**2 + (df['{0}_search_domain'.format(hyperparameters[1])].values - X_2_hit)**2
                            selidx = np.argmin(d2)
                            combination_id = df['combination_id'][selidx]
                            combination_folder = '{0}/hp_combination_{1!s}'.format(search_folder, combination_id)
                            selected_df = df[df.combination_id == combination_id].copy()
                            selected_df['search_id'] = search_id
                            selected_df['evaluation_statistic'] = evaluation_statistic
                            selected_df['selection_criterion'] = selection_criterion
                            selected_df['sigma_multiplier'] = sigma_multiplier
                            selected_df['Y_diff_hat_stdv_min'] = Y_diff_hat_stdv_min
                            selected_df['Y_diff_hat_mean_min'] = Y_diff_hat_mean_min
                            selected_df['Y_predict_hat_mean_max'] = Y_predict_hat_mean_max
                            selected_df['Y_predict_hat_stdv_max'] = Y_predict_hat_stdv_max
                            selected_df['Y_predict_hat_stdv_hit'] = Y_predict_hat_stdv[hidx]
                            selected_df['Y_predict_hat_mean_hit'] = Y_predict_hat_mean[hidx]
                            selected_df['Y_diff_hat_stdv_hit'] = Y_diff_hat_stdv[hidx]
                            selected_df['Y_diff_hat_mean_hit'] = Y_diff_hat_mean[hidx]
                            selected_df['X_1_hit'] = X_1_hit
                            selected_df['X_2_hit'] = X_2_hit
                            kernel_params = gp_predict.kernel_.get_params()
                            selected_df['kernel_noise_stdv'] = np.sqrt(kernel_params['k1__noise_level'])
                            selected_df['kernel_amplitude'] = kernel_params['k2__k1__constant_value']
                            selected_df['kernel_X_1_length_scale'], selected_df['kernel_X_2_length_scale'] = kernel_params['k2__k2__length_scale']
                            print('Y_predict_hat_mean_max: {0:1.3g}'.format(selected_df['Y_predict_hat_mean_max'].values[0]), flush=True)
                            print('Y_predict_hat_stdv_max: {0:1.3g}'.format(selected_df['Y_predict_hat_stdv_max'].values[0]), flush=True)
                            print('kernel_noise_stdv: {0:1.3g}'.format(selected_df['kernel_noise_stdv'].values[0]), flush=True)
                            print('kernel_amplitude: {0:1.3g}'.format(selected_df['kernel_amplitude'].values[0]), flush=True)
                            print('kernel_X_1_length_scale: {0:1.3g}'.format(selected_df['kernel_X_1_length_scale'].values[0]), flush=True)
                            print('kernel_X_2_length_scale: {0:1.3g}'.format(selected_df['kernel_X_2_length_scale'].values[0]), flush=True)
                            print('selected combination_id: {0!s}'.format(combination_id), flush=True)
                            print('selected combination_folder: {0}'.format(combination_folder), flush=True)
                            print('selected {0}_model_space: {1:1.3g}'.format(hyperparameters[0], selected_df['{0}_model_space'.format(hyperparameters[0])].values[0]), flush=True)
                            print('selected {0}_model_space: {1:1.3g}'.format(hyperparameters[1], selected_df['{0}_model_space'.format(hyperparameters[1])].values[0]), flush=True)
                            print('selected validation_{0}_fit: {1:1.3g}'.format(evaluation_statistic, selected_df['validation_{0}_fit'.format(evaluation_statistic)].values[0]), flush=True)
                            print('selected validation_{0}_predict: {1:1.3g}'.format(evaluation_statistic, selected_df['validation_{0}_predict'.format(evaluation_statistic)].values[0]), flush=True)
                            print('selected testing_{0}_fit: {1:1.3g}'.format(evaluation_statistic, selected_df['testing_{0}_fit'.format(evaluation_statistic)].values[0]), flush=True)
                            print('selected testing_{0}_predict: {1:1.3g}'.format(evaluation_statistic, selected_df['testing_{0}_predict'.format(evaluation_statistic)].values[0]), flush=True)
                            print('selected validation_ppv_fit: {0:1.3g}'.format(selected_df['validation_ppv_fit'].values[0]), flush=True)
                            print('selected validation_ppv_predict: {0:1.3g}'.format(selected_df['validation_ppv_predict'].values[0]), flush=True)
                            print('selected testing_ppv_fit: {0:1.3g}'.format(selected_df['testing_ppv_fit'].values[0]), flush=True)
                            print('selected testing_ppv_predict: {0:1.3g}'.format(selected_df['testing_ppv_predict'].values[0]), flush=True)
                            print('selected validation_tpr_fit: {0:1.3g}'.format(selected_df['validation_tpr_fit'].values[0]), flush=True)
                            print('selected validation_tpr_predict: {0:1.3g}'.format(selected_df['validation_tpr_predict'].values[0]), flush=True)
                            print('selected testing_tpr_fit: {0:1.3g}'.format(selected_df['testing_tpr_fit'].values[0]), flush=True)
                            print('selected testing_tpr_predict: {0:1.3g}'.format(selected_df['testing_tpr_predict'].values[0]), flush=True)
                            
                            feature_weights_path = '{0}/iter_feature_datamatrix.txt.gz'.format(combination_folder)
                            if os.path.exists(feature_weights_path) and os.path.getsize(feature_weights_path) > 0:
                                iter_feature = datasetIO.load_datamatrix(feature_weights_path)
                                iter_feature.rowmeta[iter_feature.rowname] = iter_feature.rowlabels.copy()
                                iter_feature.rowmeta['combination_id'] = selected_df['combination_id'].values.copy()
                                iter_feature.rowmeta['search_id'] = selected_df['search_id'].values.copy()
                                iter_feature.rowmeta['evaluation_statistic'] = selected_df['evaluation_statistic'].values.copy()
                                iter_feature.rowmeta['selection_criterion'] = selected_df['selection_criterion'].values.copy()
                                iter_feature.rowmeta['sigma_multiplier'] = selected_df['sigma_multiplier'].values.copy()
                                iter_feature.rowname = 'combination_id|search_id|evaluation_statistic|selection_criterion|sigma_multiplier'
                                iter_feature.rowlabels = np.array(['{0!s}|{1!s}|{2}|{3}|{4!s}'.format(ci, si, es, sc, sm) for ci, si, es, sc, sm in zip(iter_feature.rowmeta['combination_id'], iter_feature.rowmeta['search_id'], iter_feature.rowmeta['evaluation_statistic'], iter_feature.rowmeta['selection_criterion'], iter_feature.rowmeta['sigma_multiplier'])], dtype='object')
                                if 'feature_weights_dm' not in locals():
                                    feature_weights_dm = iter_feature
                                else:
                                    feature_weights_dm.append(iter_feature, 0)
                                del iter_feature
                            
                            if 'collected_df' not in locals():
                                collected_df = selected_df
                            else:
                                collected_df = collected_df.append(selected_df, ignore_index=True)
                            del selected_df
                
            else:
                print('missing combination data for search_id {0!s}. there are only {1!s} combinations'.format(search_id, df.shape[0]), flush=True)

        else:
            print('missing search data for search_id {0!s}'.format(search_id), flush=True)
                
        if np.mod(search_id, 10) == 0:
            collected_df.to_csv('{0}_selected_hyperparameters_gp_multi_4.csv'.format(project_name), index=False)
            datasetIO.save_datamatrix('{0}_selected_hyperparameters_gp_multi_feature_weights_4.txt.gz'.format(project_name), feature_weights_dm)
            
    
    collected_df.to_csv('{0}_selected_hyperparameters_gp_multi_4.csv'.format(project_name), index=False)
    datasetIO.save_datamatrix('{0}_selected_hyperparameters_gp_multi_feature_weights_4.txt.gz'.format(project_name), feature_weights_dm)

    
    print('done select_hyperparameters_gp.py', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select best hyperparameter combination and summarize.')
    parser.add_argument('--project_name', help='name of project folder', type=str)
    parser.add_argument('--hyperparameters', help='names of hyperparameters', type=str, nargs='+')
    parser.add_argument('--evaluation_statistics', help='names of performance evaluation statistics', type=str, nargs='+')
    parser.add_argument('--selection_criteria', help='heuristics for selecting hyperparameter combination', type=str, nargs='+')
    parser.add_argument('--sigma_multipliers', help='uncertainty thresholds for performance evaluation statistic', type=float, nargs='+')
    args = parser.parse_args()
    main(args.project_name, args.hyperparameters, args.evaluation_statistics, args.selection_criteria, args.sigma_multipliers)
