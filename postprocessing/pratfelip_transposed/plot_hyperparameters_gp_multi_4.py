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

import os
import argparse
import numpy as np
import pandas as pd
import datasetIO
import copy
from matplotlib import pyplot as plt
from itertools import product


def get_step_density(x, num_bins='auto', bounds='auto', apply_log10_transformation=False):
    if apply_log10_transformation:
        x = np.log10(x)
    if bounds == 'auto':
        try:
            densities, edges = np.histogram(x, num_bins, density=True)
        except MemoryError:
            densities, edges = np.histogram(x, 20, density=True)
    else:
        try:
            densities, edges = np.histogram(x, num_bins, bounds, density=True)
        except MemoryError:
            densities, edges = np.histogram(x, 20, bounds, density=True)
    densities = np.concatenate((np.append([0], densities).reshape(-1,1), np.append(densities, [0]).reshape(-1,1)), 1).reshape(-1)
    edges = np.concatenate((edges.reshape(-1,1), edges.reshape(-1,1)), 1).reshape(-1)
    return densities, edges

def plot_step_density(x, xlabel, title, save_path, num_bins='auto', bounds='auto', apply_log10_transformation=False):
    densities, edges = get_step_density(x, num_bins, bounds, apply_log10_transformation)
    fg_pdf, ax_pdf = plt.subplots(1, 1, figsize=(3,2))
    ax_pdf.plot(edges, densities, '-k', linewidth=0.5)
    ax_pdf.fill_between(edges, densities, linewidth=0, color='k', alpha=0.5)
    ax_pdf.set_position([0.55/3, 0.35/2, 2.1/3, 1.3/2]) # left, bottom, width, height
    ax_pdf.set_title(title, fontsize=8)
    ax_pdf.set_ylabel('Probability Density', fontsize=8, labelpad=4)
    ax_pdf.set_xlabel(xlabel, fontsize=8, labelpad=2)
    if bounds != 'auto':
        ax_pdf.set_xlim(bounds)
#    ax_pdf.legend(loc='upper left', ncol=1, fontsize=8, frameon=False, borderpad=1.5, labelspacing=0.1, handletextpad=0.1, borderaxespad=0)
    ax_pdf.tick_params(axis='both', which='major', bottom=True, top=False, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=8)
    ax_pdf.ticklabel_format(axis='both', style='sci', scilimits=(-3,3), fontsize=8)
    ax_pdf.yaxis.offsetText.set_fontsize(8)
    ax_pdf.xaxis.offsetText.set_fontsize(8)
    fg_pdf.savefig(save_path, transparent=True, pad_inches=0, dpi=300)
    plt.close()
    return densities, edges


def main(project_name, hyperparameters, evaluation_statistics, selection_criteria, sigma_multipliers):
    
    figures_folder = 'figures_4'
    reporting_statistics = list(sorted(set(evaluation_statistics + ['tpr', 'ppv', 'auroc', 'auprc'])))
    stages = ['validation', 'testing']
    subsets = ['fit', 'predict']
    
    reporting_fields = ['Y_predict_hat_mean_max', 'Y_predict_hat_stdv_max', 'kernel_noise_stdv', 'X_1_hit', 'X_2_hit']
    boundss = [(0,1), (0,1), (0,1), (0,1), (0,1)]
    apply_log10_transformations = [False, False, False, False, False]
    reporting_fields += ['{0}_model_space'.format(hp) for hp in hyperparameters]
    boundss += ['auto' for hp in hyperparameters]
    apply_log10_transformations += [True for hp in hyperparameters]
    reporting_fields += ['{0}_search_domain'.format(hp) for hp in hyperparameters]
    boundss += [(0,1) for hp in hyperparameters]
    apply_log10_transformations += [False for hp in hyperparameters]
    reporting_fields += ['{0}_{1}_{2}'.format(stage, reporting_statistic, subset) for stage, reporting_statistic, subset in product(stages, reporting_statistics, subsets)]
    boundss += [(0,1) for stage, reporting_statistic, subset in product(stages, reporting_statistics, subsets)]
    apply_log10_transformations += [False for stage, reporting_statistic, subset in product(stages, reporting_statistics, subsets)]
    
    if not os.path.exists('{0}/{1}'.format(figures_folder, project_name)):
        os.makedirs('{0}/{1}'.format(figures_folder, project_name))
    
    print('project: {0}...'.format(project_name), flush=True)
    print('loading data...', flush=True)
    collected_df = pd.read_csv('{0}_selected_hyperparameters_gp_multi_4.csv'.format(project_name), index_col=False)
    feature_weights_dm = datasetIO.load_datamatrix('{0}_selected_hyperparameters_gp_multi_feature_weights_4.txt.gz'.format(project_name))
    feature_weights_dm.rowmeta['sigma_multiplier'] = np.int64(feature_weights_dm.rowmeta['sigma_multiplier'])
    feature_weights_dm.rowmeta['intercept'] = np.float64(feature_weights_dm.rowmeta['intercept'])
    feature_weights_dm.rowmeta['test_index'] = np.int64(feature_weights_dm.rowmeta['test_index'])
    feature_weights_dm.rowmeta['valid_index'] = np.int64(feature_weights_dm.rowmeta['valid_index'])
    feature_weights_dm.rowmeta['combination_id'] = np.int64(feature_weights_dm.rowmeta['combination_id'])
    feature_weights_dm.rowmeta['search_id'] = np.int64(feature_weights_dm.rowmeta['search_id'])
    
    for evaluation_statistic in evaluation_statistics:
        print('working on performance evaluation statistic: {0}...'.format(evaluation_statistic), flush=True)
        hit_df_es = collected_df['evaluation_statistic'] == evaluation_statistic
        hit_dm_es = feature_weights_dm.rowmeta['evaluation_statistic'] == evaluation_statistic
        
        for selection_criterion in selection_criteria:
            print('working on selection criterion: {0}...'.format(selection_criterion), flush=True)
            hit_df_sc = collected_df['selection_criterion'] == selection_criterion
            hit_dm_sc = feature_weights_dm.rowmeta['selection_criterion'] == selection_criterion
            
            for sigma_multiplier in sigma_multipliers:
                print('working on sigma multiplier: {0}...'.format(sigma_multiplier), flush=True)
                hit_df_sm = collected_df['sigma_multiplier'] == sigma_multiplier
                hit_dm_sm = feature_weights_dm.rowmeta['sigma_multiplier'] == sigma_multiplier
                
                hit_df = np.logical_and(np.logical_and(hit_df_es, hit_df_sc), hit_df_sm)
                hit_dm = np.logical_and(np.logical_and(hit_dm_es, hit_dm_sc), hit_dm_sm)
                
                selected_df = collected_df[hit_df].copy()
                selected_fw = copy.deepcopy(feature_weights_dm)
                selected_fw.discard(~hit_dm, 0)
                
                for field, bounds, apply_log10_transformation in zip(reporting_fields, boundss, apply_log10_transformations):
                    print('working on reporting field: {0}...'.format(field), flush=True)
                    values = selected_df[field].values
                    title = '{0}, {1}, {2!s}, {3:1.3g}'.format(evaluation_statistic, selection_criterion, sigma_multiplier, np.median(values))
                    if apply_log10_transformation:
                        field = 'log10({0})'.format(field)
                    save_path = '{0}/{1}/{2}_{3}_{4}_{5!s}.png'.format(figures_folder, project_name, field, evaluation_statistic, selection_criterion, sigma_multiplier)
                    densities, edges = plot_step_density(values, field, title, save_path, 'auto', bounds, apply_log10_transformation)
                
                fg_pdf, ax_pdf = plt.subplots(1, 1, figsize=(3,2))
                ax_pdf.hist2d(selected_df['testing_tpr_predict'].values, selected_df['testing_ppv_predict'].values, bins=10, range=[[0, 1], [0, 1]], normed=True)
                ax_pdf.set_position([0.55/3, 0.35/2, 2.1/3, 1.3/2]) # left, bottom, width, height
                ax_pdf.set_title('{0}, {1}, {2!s}'.format(evaluation_statistic, selection_criterion, sigma_multiplier), fontsize=8)
                ax_pdf.set_ylabel('testing_ppv_predict', fontsize=8, labelpad=4)
                ax_pdf.set_xlabel('testing_tpr_predict', fontsize=8, labelpad=2)
            #    ax_pdf.legend(loc='upper left', ncol=1, fontsize=8, frameon=False, borderpad=1.5, labelspacing=0.1, handletextpad=0.1, borderaxespad=0)
                ax_pdf.tick_params(axis='both', which='major', bottom=True, top=False, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=8)
                ax_pdf.ticklabel_format(axis='both', style='sci', scilimits=(-3,3), fontsize=8)
                ax_pdf.yaxis.offsetText.set_fontsize(8)
                ax_pdf.xaxis.offsetText.set_fontsize(8)
                fg_pdf.savefig('{0}/{1}/{2}_VS_{3}_{4}_{5}_{6!s}.png'.format(figures_folder, project_name, 'testing_ppv_predict', 'testing_tpr_predict', evaluation_statistic, selection_criterion, sigma_multiplier), transparent=True, pad_inches=0, dpi=300)
                plt.close()
                
                # plot best features
                pcts = [0, 2.5, 5, 25, 50, 75, 95, 97.5, 100]
                ft_pct = selected_fw.totranspose()
                ft_pct.columnname = 'percentile'
                ft_pct.columnlabels = np.array([str(x) + 'PCT' for x in pcts], dtype='object')
                ft_pct.columnmeta = {'percentile':np.array(pcts, dtype='float64')}
                ft_pct.matrixname = 'feature_weight_percentiles'
                ft_pct.matrix = np.percentile(selected_fw.matrix, pcts, 0).T
                ft_pct.updatesizeattribute()
                ft_pct.updateshapeattribute()
                ft_pct.updatedtypeattribute()
                ft_pct.reorder(np.argsort(np.abs(ft_pct.matrix[:,4]))[::-1], 0)
                datasetIO.save_datamatrix('{0}/{1}/ft_pct_datamatrix_{2}_{3}_{4!s}.txt.gz'.format(figures_folder, project_name, evaluation_statistic, selection_criterion, sigma_multiplier), ft_pct)
                del ft_pct
                
                lbpct = 2.5
                ubpct = 97.5
                lb_fw = np.percentile(selected_fw.matrix, lbpct, 0)
                ub_fw = np.percentile(selected_fw.matrix, ubpct, 0)
                tobediscarded = np.logical_and(lb_fw <= 0, ub_fw >= 0)
                selected_fw.discard(tobediscarded, 1)
                median_fw = np.median(selected_fw.matrix, 0)
                sidxs = np.argsort(np.abs(median_fw))[::-1]
                bounds = 'auto'
                apply_log10_transformation = False
                for i, sidx in enumerate(sidxs):
                    field = selected_fw.columnlabels[sidx]
                    values = selected_fw.matrix[:,sidx]
                    title = '{0}, {1}, {2!s}, {3:1.3g}'.format(evaluation_statistic, selection_criterion, sigma_multiplier, np.median(values))
                    if apply_log10_transformation:
                        field = 'log10({0})'.format(field)
                    save_path = '{0}/{1}/ft{2!s}_{3}_{4}_{5}_{6!s}.png'.format(figures_folder, project_name, i, field, evaluation_statistic, selection_criterion, sigma_multiplier)
                    densities, edges = plot_step_density(values, field, title, save_path, 'auto', bounds, apply_log10_transformation)
                
                pcts = [0, 2.5, 5, 25, 50, 75, 95, 97.5, 100]
                ft_pct = selected_fw.totranspose()
                ft_pct.columnname = 'percentile'
                ft_pct.columnlabels = np.array([str(x) + 'PCT' for x in pcts], dtype='object')
                ft_pct.columnmeta = {'percentile':np.array(pcts, dtype='float64')}
                ft_pct.matrixname = 'feature_weight_percentiles'
                ft_pct.matrix = np.percentile(selected_fw.matrix, pcts, 0).T
                ft_pct.updatesizeattribute()
                ft_pct.updateshapeattribute()
                ft_pct.updatedtypeattribute()
                ft_pct.reorder(np.argsort(np.abs(ft_pct.matrix[:,4]))[::-1], 0)
                datasetIO.save_datamatrix('{0}/{1}/ft_pct_datamatrix_top_{2}_{3}_{4!s}.txt.gz'.format(figures_folder, project_name, evaluation_statistic, selection_criterion, sigma_multiplier), ft_pct)
                del ft_pct

    
    print('done plot_hyperparameters_gp_multi.py', flush=True)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select best hyperparameter combination and summarize.')
    parser.add_argument('--project_name', help='name of project folder', type=str)
    parser.add_argument('--hyperparameters', help='names of hyperparameters', type=str, nargs='+')
    parser.add_argument('--evaluation_statistics', help='names of performance evaluation statistics', type=str, nargs='+')
    parser.add_argument('--selection_criteria', help='heuristics for selecting hyperparameter combination', type=str, nargs='+')
    parser.add_argument('--sigma_multipliers', help='uncertainty thresholds for performance evaluation statistic', type=float, nargs='+')
    args = parser.parse_args()
    main(args.project_name, args.hyperparameters, args.evaluation_statistics, args.selection_criteria, args.sigma_multipliers)
