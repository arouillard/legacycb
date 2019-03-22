# -*- coding: utf-8 -*-
"""
@author: ar988996
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import datasetIO
import dataclasses
import copy
import sys


def relu(mat):
    mat[mat < 0] = 0
    return mat

def sdae_transform(dm, W, Be, activation, apply_activation_to_output=False):
    mat = dm.matrix.copy()
    for i, (w, b) in enumerate(zip(W, Be)):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w) + b)
        else:
            mat = mat.dot(w) + b
    em = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=copy.deepcopy(dm.rowmeta),
                                columnname='latent_component',
                                columnlabels=np.array(['LC'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                columnmeta={'activation_applied':np.full(mat.shape[1], apply_activation_to_output, dtype='bool')},
                                matrixname='sdae_transform_of_'+dm.matrixname,
                                matrix=mat)
    return em

def sdae_inverse_transform(em, W, Bd, activation, apply_activation_to_output=False):
    if ~(em.columnmeta['activation_applied'].any()):
        mat = activation(em.matrix)
    else:
        mat = em.matrix.copy()
    for i, (w, b) in enumerate(zip(W[::-1], Bd[::-1])):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w.T) + b)
        else:
            mat = mat.dot(w.T) + b
    rm = dataclasses.datamatrix(rowname=em.rowname,
                                rowlabels=em.rowlabels.copy(),
                                rowmeta=copy.deepcopy(em.rowmeta),
                                columnname='reconstructed_feature',
                                columnlabels=np.array(['RF'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                columnmeta={},
                                matrixname='reconstruction_from_'+em.matrixname,
                                matrix=mat)
    return rm

def sdae_reconstruction(dm, W, Be, Bd, activation, apply_activation_to_output=False, return_embedding=False, return_reconstruction_error=False):
    mat = dm.matrix.copy()
    for i, (w, b) in enumerate(zip(W, Be)):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w) + b)
        else:
            mat = mat.dot(w) + b
    if return_embedding:
        em = dataclasses.datamatrix(rowname=dm.rowname,
                                    rowlabels=dm.rowlabels.copy(),
                                    rowmeta=copy.deepcopy(dm.rowmeta),
                                    columnname='latent_component',
                                    columnlabels=np.array(['LC'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                    columnmeta={'activation_applied':np.full(mat.shape[1], apply_activation_to_output, dtype='bool')},
                                    matrixname='sdae_transform_of_'+dm.matrixname,
                                    matrix=mat.copy())
    if not apply_activation_to_output:
        mat = activation(mat)
    for i, (w, b) in enumerate(zip(W[::-1], Bd[::-1])):
        if i+1 < len(W) or apply_activation_to_output:
            mat = activation(mat.dot(w.T) + b)
        else:
            mat = mat.dot(w.T) + b
    rm = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=copy.deepcopy(dm.rowmeta),
                                columnname='reconstructed_' + dm.columnname,
                                columnlabels=dm.columnlabels.copy(),
                                columnmeta=copy.deepcopy(dm.columnmeta),
                                matrixname='reconstruction_from_sdae_transform_of_'+dm.matrixname,
                                matrix=mat)
    reconstruction_error = np.mean((rm.matrix - dm.matrix)**2)
    if return_embedding and return_reconstruction_error:
        return rm, em, reconstruction_error
    elif return_embedding:
        return rm, em
    elif return_reconstruction_error:
        return rm, reconstruction_error
    else:
        return rm


def main(study_name='your_study'):
    
    # load the data
    orientation = 'fat'
    partitions = ['train', 'valid', 'test']
    
    dataset = {}
    for partition in partitions:
        dataset[partition] = datasetIO.load_datamatrix('data/prepared_data/{0}/{1}.pickle'.format(orientation, partition))
        if 'all' not in dataset:
            dataset['all'] = copy.deepcopy(dataset[partition])
        else:
            dataset['all'].append(dataset[partition], 0)
    
    dataset[study_name] = {}
    for partition in partitions:
        dataset[study_name][partition] = datasetIO.load_datamatrix('data/prepared_data/{0}/{1}/{2}.pickle'.format(study_name, orientation, partition))
        if 'all' not in dataset[study_name]:
            dataset[study_name]['all'] = copy.deepcopy(dataset[study_name][partition])
        else:
            dataset[study_name]['all'].append(dataset[study_name][partition], 0)
    
    partitions.append('all')
    
    
    # create output directories
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/sdae_features'):
        os.mkdir('results/sdae_features')
    if not os.path.exists('results/sdae_features/{0}'.format(study_name)):
        os.mkdir('results/sdae_features/{0}'.format(study_name))
    if not os.path.exists('results/sdae_features/{0}/{1}'.format(study_name, orientation)):
        os.mkdir('results/sdae_features/{0}/{1}'.format(study_name, orientation))
    
    
    # load the model
    activation_function, activation_function_name = (relu, 'relu')
    with open('results/autoencoder/fat/ns5_last2_first0.05_5layers_relu_variables.pickle', 'rb') as fr:
        W, Be, Bd = pickle.load(fr)[1:] # global_step, W, bencode, bdecode
    
    
    # get embeddings and reconstructions
    sdae = {}
    for partition in partitions:
        sdae[partition] = {}
        sdae[partition]['recon'], sdae[partition]['embed'], sdae[partition]['error'] = sdae_reconstruction(dataset[partition], W, Be, Bd, activation=activation_function, apply_activation_to_output=False, return_embedding=True, return_reconstruction_error=True)
        print('{0} error: {1:1.3g}'.format(partition, sdae[partition]['error']))
    
    sdae[study_name] = {}
    for partition in partitions:
        sdae[study_name][partition] = {}
        sdae[study_name][partition]['recon'], sdae[study_name][partition]['embed'], sdae[study_name][partition]['error'] = sdae_reconstruction(dataset[study_name][partition], W, Be, Bd, activation=activation_function, apply_activation_to_output=False, return_embedding=True, return_reconstruction_error=True)
        print('{0} {1} error: {2:1.3g}'.format(study_name, partition, sdae[study_name][partition]['error']))
    
    
    # visualize embedding
    if sdae['all']['embed'].shape[1] < 5:
        for nx in range(sdae['all']['embed'].shape[1]-1):
            for ny in range(nx+1, sdae['all']['embed'].shape[1]):
                
                #tissues = np.unique(dataset['all'].rowmeta['general_tissue'])
                tissues = ['Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel', 'Brain',
                           'Breast', 'Colon', 'Esophagus', 'Heart', 'Kidney', 'Liver', 'Lung', 'Muscle',
                           'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Salivary Gland', 'Skin',
                           'Small Intestine', 'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Uterus', 'Vagina']
                tissue_abbrevs = ['AT', 'AG', 'B', 'BV', 'Bn',
                                  'Bt', 'C', 'E', 'H', 'K', 'Lr', 'Lg', 'M',
                                  'N', 'O', 'Ps', 'Py', 'Pe', 'SG', 'Sk',
                                  'SI', 'Sp', 'St', 'Ts', 'Td', 'U', 'V']
                cmap = plt.get_cmap('gist_rainbow')
                colors = [cmap(float((i+0.5)/len(tissues))) for i in range(len(tissues))]
                
                fg, ax = plt.subplots(1, 1, figsize=(6.5,4.3))
                ax.set_position([0.15/6.5, 0.15/4.3, 4.0/6.5, 4.0/4.3])
                for tissue, tissue_abbrev, color in zip(tissues, tissue_abbrevs, colors):
                    if tissue == '-666':
                        continue
                #        zorder = 0
                #        alpha = 0.05
                #        color = 'k'
                    else:
                        zorder = 1
                        alpha = 0.5
                    hit = dataset['all'].rowmeta['general_tissue'] == tissue
                    hidxs = hit.nonzero()[0]
                #    ax.plot(sdae['all']['embed'].matrix[hit,nx], sdae['all']['embed'].matrix[hit,ny], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=2, markeredgewidth=0, alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                    ax.plot(sdae['all']['embed'].matrix[hit,nx], sdae['all']['embed'].matrix[hit,ny], linestyle='None', linewidth=0, marker='o', markerfacecolor=color, markeredgecolor=color, markersize=0.2, markeredgewidth=0, alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                    for hidx in hidxs:
                        ax.text(sdae['all']['embed'].matrix[hidx,nx], sdae['all']['embed'].matrix[hidx,ny], tissue_abbrev, horizontalalignment='center', verticalalignment='center', fontsize=4, color=color, alpha=alpha, zorder=zorder, label='{0}, {1}'.format(tissue_abbrev, tissue))
                ax.plot(sdae[study_name]['all']['embed'].matrix[:,nx], sdae[study_name]['all']['embed'].matrix[:,ny], linestyle='None', linewidth=0, marker='x', markerfacecolor='k', markeredgecolor='k', markersize=0.2, markeredgewidth=0, alpha=1, zorder=1, label=study_name)
                for hidx in range(sdae[study_name]['all']['embed'].shape[0]):
                    ax.text(sdae[study_name]['all']['embed'].matrix[hidx,nx], sdae[study_name]['all']['embed'].matrix[hidx,ny], 'X', horizontalalignment='center', verticalalignment='center', fontsize=4, color='k', alpha=1, zorder=1, label=study_name)
                ax.set_xlim(sdae['all']['embed'].matrix[:,nx].min(), sdae['all']['embed'].matrix[:,nx].max())
                ax.set_ylim(sdae['all']['embed'].matrix[:,ny].min(), sdae['all']['embed'].matrix[:,ny].max())
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0, frameon=False, ncol=1, numpoints=1, markerscale=40, fontsize=8, labelspacing=0.25)
                ax.tick_params(axis='both', which='major', bottom='off', top='off', labelbottom='off', labeltop='off', left='off', right='off', labelleft='off', labelright='off', pad=4)
                ax.set_frame_on(False)
                fg.savefig('results/sdae_features/{0}/{1}/sdae2d_{2}_coloredby_general_tissue_x{3!s}_y{4!s}.png'.format(study_name, orientation, activation_function_name, nx, ny), transparent=True, pad_inches=0, dpi=600)
                ax.set_xlim(sdae[study_name]['all']['embed'].matrix[:,nx].min(), sdae[study_name]['all']['embed'].matrix[:,nx].max())
                ax.set_ylim(sdae[study_name]['all']['embed'].matrix[:,ny].min(), sdae[study_name]['all']['embed'].matrix[:,ny].max())
                fg.savefig('results/sdae_features/{0}/{1}/sdae2d_{2}_coloredby_general_tissue_x{3!s}_y{4!s}_zoom.png'.format(study_name, orientation, activation_function_name, nx, ny), transparent=True, pad_inches=0, dpi=600)
                plt.close()
    
    
    # save embedding
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae2d_{2}_datamatrix.txt.gz'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['embed'])
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae2d_{2}_datamatrix.pickle'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['embed'])
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae_reconstructions_{2}_datamatrix.txt.gz'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['recon'])
    datasetIO.save_datamatrix('results/sdae_features/{0}/{1}/sdae_reconstructions_{2}_datamatrix.pickle'.format(study_name, orientation, activation_function_name), sdae[study_name]['all']['recon'])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise ValueError('too many inputs')

