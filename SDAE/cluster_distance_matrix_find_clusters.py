# -*- coding: utf-8 -*-
"""
Clinical Outcome Classifier
get scores for individual features
multiple hypothesis correction by dataset or altogether?
what is the best test?
"""

import sys
#custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
for custompath in custompaths:
    if custompath not in sys.path:
        sys.path.append(custompath)
del custompath, custompaths

import os
#import gzip
import numpy as np
import copy
from machinelearning import datasetselection, featureselection
import machinelearning.dataclasses as dc
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import fastcluster
from sklearn.metrics import silhouette_score, calinski_harabaz_score







D = distance.squareform(np.float64(1) - ind_ind.matrix, checks=False)
Z = fastcluster.linkage(D, 'average')
D = distance.squareform(D)
numclusters = np.arange(100, ind_ind.shape[0]-1, 100, dtype='int64')
print(numclusters.size)
silhouette = np.zeros_like(numclusters, dtype='float64')
for i, nc in enumerate(numclusters):
    print('working on numclusters={0!s}'.format(nc))
    C = hierarchy.cut_tree(Z, nc).reshape(-1)
    silhouette[i] = silhouette_score(D, C, 'precomputed')
selectednumclusters = numclusters[silhouette == silhouette[~np.isnan(silhouette)].max()][0]

ncwindows = [500, 250, 100, 50, 10]
ncsteps = [50, 25, 10, 5, 1]
for ncwindow, ncstep in zip(ncwindows, ncsteps):
    numclusters_fine = np.arange(selectednumclusters-ncwindow, selectednumclusters+ncwindow, ncstep, dtype='int64')
    numclusters_fine = numclusters_fine[~np.in1d(numclusters_fine, numclusters)]
    print(numclusters_fine.size)
    silhouette_fine = np.zeros_like(numclusters_fine, dtype='float64')
    for i, nc in enumerate(numclusters_fine):
        print('working on numclusters_fine={0!s}'.format(nc))
        C = hierarchy.cut_tree(Z, nc).reshape(-1)
        silhouette_fine[i] = silhouette_score(D, C, 'precomputed')
    numclusters = np.append(numclusters, numclusters_fine)
    silhouette = np.append(silhouette, silhouette_fine)
    si = np.argsort(numclusters)
    numclusters = numclusters[si]
    silhouette = silhouette[si]
    selectednumclusters = numclusters[silhouette == silhouette[~np.isnan(silhouette)].max()][0]
with open('indication_indication_mesh_termite_cluster_quality.pickle', 'wb') as fw:
    pickle.dump({'silhouette':silhouette, 'numclusters':numclusters, 'selectednumclusters':selectednumclusters}, fw)

fig = plt.figure(figsize=(3,2))
ax = fig.add_subplot(111)
ax.set_position([0.5/3, 0.4/2, 2.35/3, 1.35/2])
ax.plot(numclusters, silhouette, '-k')
ax.set_xlabel('Clusters', fontsize=8, fontname='arial')
ax.set_ylabel('Silhouette Score', fontsize=8, fontname='arial')
ax.set_title('Cluster Quality', fontsize=8, fontname='arial')
ax.tick_params(axis='both', which='major', bottom='on', top='off', left='on', right='off', labelbottom='on', labeltop='off', labelleft='on', labelright='off', labelsize=8)
ax.set_xlim(0, ind_ind.shape[0])
ax.set_ylim(0, 1)
plt.savefig('silhouette_analysis.png', transparent=True, pad_inches=0, dpi=1200)
plt.show()

ind_ind.rowmeta['cluster'] = hierarchy.cut_tree(Z, selectednumclusters).reshape(-1)
ind_ind.rowmeta['clustered_order'] = hierarchy.leaves_list(Z).astype('int64')
ind_ind.columnmeta = copy.deepcopy(ind_ind.rowmeta)
ind_ind.reorder(ind_ind.rowmeta['clustered_order'].copy(), 0)
ind_ind.reorder(ind_ind.columnmeta['clustered_order'].copy(), 1)
ind_ind.heatmap([], [], savefilename='indication_indication_mesh_termite.png')
plt.savefig('indication_indication_mesh_termite_highres.png', transparent=True, pad_inches=0, dpi=1200)
with open('indication_indication_mesh_termite_with_clusters.pickle', 'wb') as fw:
    pickle.dump(ind_ind, fw)








def silhouette_scr(D, C):
    unique_clusters, cluster_sizes = np.unique(C, return_counts=True)
    C[np.in1d(C, unique_clusters[cluster_sizes==1])] = unique_clusters.max() + 1 # this changes C outside the function
    unique_clusters = np.unique(C)
    intra_cluster_dists = np.zeros(D.shape[0], dtype='float64')
    inter_cluster_dists = np.full(D.shape[0], np.inf, dtype='float64')
    for i, ci in enumerate(unique_clusters):
        hi = C==ci
        Dci = D[hi,:]
        intra_cluster_dists[hi] = Dci[:,hi].sum(1)/(hi.sum()-1)
        for cj in unique_clusters[unique_clusters != ci]:
            inter_cluster_dists[hi] = np.min(np.append(Dci[:,C==cj].mean(1).reshape(-1,1), inter_cluster_dists[hi].reshape(-1,1), 1), 1)
    s = (inter_cluster_dists - intra_cluster_dists)/np.max(np.append(inter_cluster_dists.reshape(-1,1), intra_cluster_dists.reshape(-1,1), 1), 1)
    s[C==unique_clusters.max()] = 0
    return s.mean()

def dbi(X, cluster_labels, metric='angular_cosine'):
    if metric == 'angular_cosine':
        unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        C = cluster_labels
        C[np.in1d(C, unique_clusters[cluster_sizes==1])] = unique_clusters.max() + 1
        unique_clusters = np.unique(C)
        centroids = np.zeros((unique_clusters.size, X.shape[1]), dtype='float64')
        for i, ci in enumerate(unique_clusters):
            centroids[i,:] = X[C==ci,:].mean(0)
        sample_mags = np.sqrt((X**2).sum(1, keepdims=True))
        centroid_mags = np.sqrt((centroids**2).sum(1, keepdims=True))
        sample_centroid_dists = (X/sample_mags).dot((centroids/centroid_mags).T)
        sample_centroid_dists[sample_centroid_dists<-1] = -1
        sample_centroid_dists[sample_centroid_dists>1] = 1
        if sample_centroid_dists.min() < 0:
            sample_centroid_dists = np.arccos(sample_centroid_dists)/np.pi
        else:
            sample_centroid_dists = np.arccos(sample_centroid_dists)/(np.pi/2)
        centroid_centroid_dists = (centroids/centroid_mags).dot((centroids/centroid_mags).T)
        centroid_centroid_dists[centroid_centroid_dists<-1] = -1
        centroid_centroid_dists[centroid_centroid_dists>1] = 1
        if centroid_centroid_dists.min() < 0:
            centroid_centroid_dists = np.arccos(centroid_centroid_dists)/np.pi
        else:
            centroid_centroid_dists = np.arccos(centroid_centroid_dists)/(np.pi/2)
        cluster_centroid_dists = np.zeros((unique_clusters.size, 1), dtype='float64')
        for i, ci in enumerate(unique_clusters):
            cluster_centroid_dists[i] = sample_centroid_dists[C==ci,i].mean()
        D = cluster_centroid_dists.dot(cluster_centroid_dists.T)/centroid_centroid_dists
        D[np.eye(unique_clusters.size, dtype='bool')] = 0
        return D.max(1).mean()
    else:
        raise ValueError('unsupported distance metric')
    
        

# load distance matrix
with open('gene_gene_matrix_angular_cosine_distance_from_zscored_tissue_expression.pickle', 'rb') as fr:
    gene_gene = pickle.load(fr)
with open('gene_atb_matrix_zscored_tissue_expression.pickle', 'rb') as fr:
    gene_atb = pickle.load(fr)

lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'average')
plt.figure(); hierarchy.dendrogram(lnk)

si = hierarchy.leaves_list(lnk).astype('int64')
ordered_genes = gene_gene.rowlabels[si]
with open('ordered_genes_angular_cosine_distance_from_zscored_tissue_expression.pickle', 'wb') as fw:
    pickle.dump(ordered_genes, fw)

numclusters = np.concatenate((np.arange(2, 100, 1, dtype='int64'), np.arange(100, 1000, 10, dtype='int64'), np.arange(1000, 10000, 100, dtype='int64')))
silhouette = np.zeros_like(numclusters, dtype='float64')
chs = np.zeros((numclusters.size, 2), dtype='float64')
for i, nc in enumerate(numclusters):
    print('working on numclusters={0!s}'.format(nc))
    cluster_labels = hierarchy.cut_tree(lnk, n_clusters=nc).reshape(-1)
    chs[i,0] = calinski_harabaz_score(gene_atb.matrix, cluster_labels)
    unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    cluster_labels[np.in1d(cluster_labels, unique_clusters[cluster_sizes==1])] = unique_clusters.max() + 1
    chs[i,1] = calinski_harabaz_score(gene_atb.matrix, cluster_labels)
    print(chs[i,:])
#    if np.mod(nc,500) == 0 and i > 232:
#        print('working on numclusters={0!s}'.format(nc))
#        cluster_labels = hierarchy.cut_tree(lnk, n_clusters=nc).reshape(-1)
#        silhouette[i] = silhouette_score(gene_gene.matrix, cluster_labels, 'precomputed')
#        print(silhouette[i])
plt.figure(); plt.plot(numclusters, silhouette/silhouette.max(), '-k', numclusters, chs[:,0]/chs[:,0].max(), ':b'); plt.xlim(0,200)
plt.figure(); plt.semilogx(numclusters, silhouette, '-k'); plt.xlim(0,1000)
plt.figure(); plt.semilogx(numclusters, chs[:,0], '-k'); plt.xlim(0,1000)
selectednumclusters = numclusters[np.argmax(silhouette)]



cluster_labels = hierarchy.cut_tree(lnk, n_clusters=7).reshape(-1)
unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
delta = np.zeros((unique_clusters.size, 6), dtype='float64')
for i, ci in enumerate(unique_clusters):
    d = gene_gene.matrix[cluster_labels==ci,:][:,cluster_labels==ci]
    delta[i,:] = [np.mean(d)/2, np.median(d)/2, np.percentile(d, 85)/2, np.percentile(d, 90)/2, np.percentile(d, 95)/2, np.percentile(d, 97.5)/2]
    plt.figure()
    plt.hist(d.reshape(-1), min(100,cluster_sizes[i]))


raise ValueError('stop')

si = {}
cst = {}
numcst = 87
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'average')
plt.figure(); hierarchy.dendrogram(lnk)
si['average'] = hierarchy.leaves_list(lnk)
cst['average'] = hierarchy.cut_tree(lnk, n_clusters=numcst).reshape(-1)
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'centroid')
plt.figure(); hierarchy.dendrogram(lnk)
si['centroid'] = hierarchy.leaves_list(lnk)
cst['centroid'] = hierarchy.cut_tree(lnk, n_clusters=285).reshape(-1) # numcst).reshape(-1) # does not return the correct number of clusters when for centroid linkage
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'ward')
plt.figure(); hierarchy.dendrogram(lnk)
si['ward'] = hierarchy.leaves_list(lnk)
cst['ward'] = hierarchy.cut_tree(lnk, n_clusters=numcst).reshape(-1)
# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)
if ~(gene_proj.rowlabels == gene_gene.rowlabels).all():
    raise ValueError('genes not aligned')
# evaluate the linkage methods
# ward appears to be best, average pretty good, centroid terrible (but may be because something is wrong with the code)
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(float((i+0.5)/numcst)) for i in range(numcst)]
for linkage_method, cluster_ids in cst.items():
    fg, ax = plt.subplots(1, 1, figsize=(6.5,5.3))
    ax.set_position([0.15/6.5, 0.05/5.3, 5.0/6.5, 4.9/5.3])
    for cluster_id, color in enumerate(colors):
        hit = cluster_ids == cluster_id
        ax.plot(gene_proj.matrix[hit,0], gene_proj.matrix[hit,1], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=0.5, zorder=1, label=str(cluster_id))
    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    ax.set_title(linkage_method, fontsize=8)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, numpoints=1, ncol=3, columnspacing=0, markerscale=4, fontsize=8, labelspacing=0.4, handletextpad=0)
    ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
    ax.set_frame_on(False)
    fg.savefig('gene_atb_matrix_2d_dnn_projection_{0!s}clusters_{1}linkage.png'.format(numcst, linkage_method), transparent=True, pad_inches=0, dpi=600)
    fg.show()
'''
# prefer ward linkage for euclidean distance or at least this case
lnk = fastcluster.linkage(distance.squareform(gene_gene.matrix, checks=False), 'ward')
#plt.figure()
#hierarchy.dendrogram(lnk)
si = hierarchy.leaves_list(lnk).astype('int64')
#cst = hierarchy.cut_tree(lnk, n_clusters=87).reshape(-1)

# load projection
with open('gene_atb_matrix_2d_dnn_projection.pickle', 'rb') as fr:
    gene_proj = pickle.load(fr)
if ~(gene_proj.rowlabels == gene_gene.rowlabels).all():
    raise ValueError('genes not aligned')
gene_proj.reorder(si, 0)
#gene_gene.reorder(si, 0)
#gene_gene.reorder(si, 1)
ordered_genes = gene_proj.rowlabels.copy()
del gene_gene, lnk, si

# select datasets
dataset_info = datasetselection.finddatasets(getalllevels=True)
included_datasetabbrevs = {'clinvar', 'dbgap_cleaned', 'gad', 'gadhighlevel_cleaned', 'gobp', 'gocc', 'gomf', 'gwascatalog_cleaned', 'gwasdbdisease_cleaned', 'gwasdbphenotype_cleaned', 'hpo', 'hugenavigator', 'humancyc', 'kegg', 'locate', 'locatepredicted', 'mgimpo', 'omim', 'panther', 'reactome', 'wikipathways'}
excluded_datasetabbrevs = set(dataset_info.keys()).difference(included_datasetabbrevs)
for datasetabbrev in excluded_datasetabbrevs:
    del dataset_info[datasetabbrev]
dataset_info['gtex_overexpression_nonspecific'] = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_nonspecific', 'name':'Tissue Over-expression 10-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_nonspecific.pickle'}
dataset_info['gtex_overexpression_toptissue'] = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_toptissue', 'name':'Top Tissue Over-expression 10-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_toptissue.pickle'}
dataset_info['gtex_overexpression_specific'] = {'folder':'gtextissue', 'abbreviation':'gtex_overexpression_specific', 'name':'Specific Tissue Over-expression 10-fold from GTEx Tissue Gene Expression Profiles - Cleaned', 'path':'C:/Users/ar988996/Documents/Harmonizome/datasets/gtextissue/gene_tissue_6xoverexpression_3xspecific.pickle'}
    
# parameters
feature_selection_test_name = 'permutation'
analysis_version = 'v7chisq14network'
results_layers = {'test_statistic_values', 'pvalues', 'correlation_sign', 'is_significant'}

# iterate over datasets
for datasetabbrev, datasetinfo in dataset_info.items():
    # just work with pathways for testing/debugging the pipeline
    if 'gtex_overexpression' not in datasetabbrev: # datasetabbrev not in {'kegg', 'panther', 'reactome', 'wikipathways'}:
        continue
    if not os.path.exists('univariate_feature_importance_{0}/{1}_pvalues.pickle'.format(analysis_version, datasetabbrev)):
        continue
    print('working on {0}...'.format(datasetabbrev))
    # load aligned matrices
#    print('loading aligned matrices...')
#    with open('aligned_matrices_{0}/gene_atb_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
#        gene_atb = pickle.load(fr)
#    background_size = gene_atb.shape[0]
#    del gene_atb
#    with open('aligned_matrices_{0}/gene_cst_{1}.pickle'.format(analysis_version, datasetabbrev), mode='rb') as fr:
#        gene_cst = pickle.load(fr)
    # load pvalues, test statistic values, etc.
    print('loading results...')
    atb_cst = {}
    for layer in results_layers:
        with open('univariate_feature_importance_{0}/{1}_{2}.pickle'.format(analysis_version, datasetabbrev, layer), mode='rb') as fr:
            atb_cst[layer] = pickle.load(fr)
    minp = atb_cst['pvalues'].matrix[atb_cst['pvalues'].matrix > 0].min()
    atb_cst['pvalues'].matrix[atb_cst['pvalues'].matrix == 0] = minp
    # keep only significant features (include effect size cutoff)
    atb_cst['is_significant'].matrix = np.logical_and(atb_cst['is_significant'].matrix, atb_cst['test_statistic_values'].matrix > 0.01)
    tobediscarded = ~(atb_cst['is_significant'].matrix.any(1))
    for layer in results_layers:
        atb_cst[layer].discard(tobediscarded, 0)
    # order columns (genes) by projection clustering and rows (features) by correlation of signed negative log10 pvalue maps
    atb_cst['snlp'] = copy.deepcopy(atb_cst['pvalues'])
    atb_cst['snlp'].matrix = -1.0*atb_cst['correlation_sign'].matrix*np.log10(atb_cst['pvalues'].matrix)
    gene_idx = {g:i for i,g in enumerate(atb_cst['snlp'].columnlabels)}
    si = np.array([gene_idx[g] for g in ordered_genes if g in gene_idx], dtype='int64')
    atb_cst['snlp'].reorder(si, 1)
    atb_cst['snlp'].cluster(0, metric='correlation', method='average')
    for layer in results_layers:
        atb_cst[layer] = atb_cst[layer].tolabels(rowlabels=atb_cst['snlp'].rowlabels.copy(), columnlabels=atb_cst['snlp'].columnlabels.copy())
    # plot results
    print('plotting results...')
#    atb_cst['snlp'].heatmap([], [], normalize='rows', savefilename='univariate_feature_importance_{0}/{1}_atb_cst_snlp_heatmap.png'.format(analysis_version, datasetabbrev), closefigure=True, dpi=1200)
#    atb_cst['test_statistic_values'].heatmap([], [], savefilename='univariate_feature_importance_{0}/{1}_atb_cst_test_statistic_values_heatmap.png'.format(analysis_version, datasetabbrev), closefigure=True, dpi=1200)
    gp_in = gene_proj.tolabels(rowlabels=atb_cst['pvalues'].columnlabels.copy())
    gp_out = gene_proj.tolabels(rowlabels=gene_proj.rowlabels[~np.in1d(gene_proj.rowlabels, gp_in.rowlabels)])
    num_sig_atbs = atb_cst['is_significant'].matrix.any(1).sum()
#    num_fig_rows = int(np.ceil(float(num_sig_atbs)/2.0))
    rows = 4
    cols = 8
    yub = 0.5
    ylb = -0.35
    xub = 0.45
    xlb = -0.4
    fg, axs = plt.subplots(rows, cols, figsize=(10,5))
    fg.subplots_adjust(left=0.025, bottom=0.025, right=0.975, top=0.9, wspace=0.2, hspace=0.2)
    fg.suptitle('{0}, {1!s} sig features'.format(datasetinfo['name'], num_sig_atbs), fontsize=8)
    axs = axs.reshape(-1)
    num_axs = rows*cols
    ia = 0
    for feature, tvalues, pvalues, snlps, signs, significances in zip(atb_cst['pvalues'].rowlabels, atb_cst['test_statistic_values'].matrix, atb_cst['pvalues'].matrix, atb_cst['snlp'].matrix, atb_cst['correlation_sign'].matrix, atb_cst['is_significant'].matrix):
        if ia >= num_axs:
            break
        if significances.any():
            ax = axs[ia]
#            cub = np.max(np.abs(snlps))
#            clb = -cub
            cub = snlps.max()
            clb = snlps.min()
            ax.scatter(gp_in.matrix[:,0], gp_in.matrix[:,1], s=0.1, c=snlps, marker='o', edgecolors='none', cmap=plt.get_cmap('jet'), vmin=clb, vmax=cub)
            ax.scatter(gp_out.matrix[:,0], gp_out.matrix[:,1], s=0.1, c='k', alpha=0.01, marker='o', edgecolors='none')
            ax.set_title(feature[:35], fontsize=4)
            ax.set_ylim(ylb, yub)
            ax.set_xlim(xlb, xub)
            ax.tick_params(axis='both', which='major', left='off', right='off', bottom='off', top='off', labelleft='off', labelright='off', labelbottom='off', labeltop='off', pad=4)
            ax.set_frame_on(False)
            ax.axvline(xlb, linewidth=0.5, color='k')
            ax.axvline(xub, linewidth=0.5, color='k')
            ax.axhline(ylb, linewidth=0.5, color='k')
            ax.axhline(yub, linewidth=0.5, color='k')
            ia += 1
    if ia < num_axs:
        for j in range(ia, num_axs):
            plt.delaxes(axs[j])
    fg.savefig('univariate_feature_importance_{0}/{1}_maps.png'.format(analysis_version, datasetabbrev), transparent=True, pad_inches=0, dpi=1200)
#    fg.show()
    plt.close()
print('done.')
