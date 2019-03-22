# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import numpy as np
import dataclasses
import copy

# if apply_activation_to_embedding=True, SOFTMAX is used for embedding

def relu(mat):
    mat[mat < 0] = 0
    return mat

def sigmoid(mat):
    return 1/(1 + np.exp(-mat))

def tanh(mat):
    return np.tanh(mat)

def elu(mat):
    mat[mat < 0] = np.exp(mat[mat < 0]) - 1
    return mat

def softmax(mat):
    return np.exp(mat)/np.sum(np.exp(mat), axis=-1, keepdims=True)

def align_batchnorm_variables(bn_variables, apply_activation_to_embedding, apply_activation_to_output):
    gammas, betas, moving_means, moving_variances = bn_variables
    n = int(len(betas)/2)
    if gammas == []:
        gammas = [1.0 for x in range(len(betas))]
    else:
        if not apply_activation_to_embedding:
            gammas.insert(n-1, 1.0)
        if not apply_activation_to_output:
            gammas.insert(len(gammas), 1.0)
    bn_encode_variables = (gammas[:n], betas[:n], moving_means[:n], moving_variances[:n])
    bn_decode_variables = (gammas[n:], betas[n:], moving_means[n:], moving_variances[n:])
    return bn_encode_variables, bn_decode_variables
      
def batchnorm(mat, gamma, beta, moving_mean, moving_variance, epsilon=0.001):
    return gamma*(mat - moving_mean)/np.sqrt(moving_variance + epsilon) + beta

def encode(dm, W, Be, activation, apply_activation_to_embedding=False, use_softmax=False, bn_variables=None):
    mat = dm.matrix.copy()
    if bn_variables == None:
        for i, (w, b) in enumerate(zip(W, Be)):
            if i+1 < len(W):
                mat = activation(mat.dot(w) + b)
            else:
                if apply_activation_to_embedding:
                    if use_softmax:
                        mat = softmax(mat.dot(w) + b)
                    else:
                        mat = activation(mat.dot(w) + b)
                else:
                    mat = mat.dot(w) + b
    else:
        gammas, betas, moving_means, moving_variances = bn_variables
        for i, (w, b, gamma, beta, moving_mean, moving_variance) in enumerate(zip(W, Be, gammas, betas, moving_means, moving_variances)):
            if i+1 < len(W) or apply_activation_to_embedding:
                mat = activation(batchnorm(mat.dot(w), gamma, beta, moving_mean, moving_variance))
            else:
                mat = batchnorm(mat.dot(w), gamma, beta, moving_mean, moving_variance)
    em = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=copy.deepcopy(dm.rowmeta),
                                columnname='latent_component',
                                columnlabels=np.array(['LC'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                columnmeta={},
                                matrixname='sdae_encoding_of_'+dm.matrixname,
                                matrix=mat)
    return em

def decode(em, W, Bd, activation, apply_activation_to_output=False, bn_variables=None):
    mat = em.matrix.copy()
    if bn_variables == None:
        for i, (w, b) in enumerate(zip(W[::-1], Bd[::-1])):
            if i+1 < len(W) or apply_activation_to_output:
                mat = activation(mat.dot(w.T) + b)
            else:
                mat = mat.dot(w.T) + b
    else:
        gammas, betas, moving_means, moving_variances = bn_variables
        for i, (w, b, gamma, beta, moving_mean, moving_variance) in enumerate(zip(W[::-1], Bd[::-1], gammas, betas, moving_means, moving_variances)):
            if i+1 < len(W) or apply_activation_to_output:
                mat = activation(batchnorm(mat.dot(w.T), gamma, beta, moving_mean, moving_variance))
            else:
                mat = batchnorm(mat.dot(w.T), gamma, beta, moving_mean, moving_variance)
    rm = dataclasses.datamatrix(rowname=em.rowname,
                                rowlabels=em.rowlabels.copy(),
                                rowmeta=copy.deepcopy(em.rowmeta),
                                columnname='reconstructed_feature',
                                columnlabels=np.array(['RF'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                columnmeta={},
                                matrixname='decoding_from_'+em.matrixname,
                                matrix=mat)
    return rm

def encode_and_decode(dm, W, Be, Bd, activation, apply_activation_to_embedding=False, use_softmax=False, apply_activation_to_output=False, return_embedding=False, return_reconstruction_error=False, bn_encode_variables=None, bn_decode_variables=None):
    mat = dm.matrix.copy()
    if bn_encode_variables == None:
        for i, (w, b) in enumerate(zip(W, Be)):
            if i+1 < len(W):
                mat = activation(mat.dot(w) + b)
            else:
                if apply_activation_to_embedding:
                    if use_softmax:
                        mat = softmax(mat.dot(w) + b)
                    else:
                        mat = activation(mat.dot(w) + b)
                else:
                    mat = mat.dot(w) + b
    else:
        gammas, betas, moving_means, moving_variances = bn_encode_variables
        for i, (w, b, gamma, beta, moving_mean, moving_variance) in enumerate(zip(W, Be, gammas, betas, moving_means, moving_variances)):
            if i+1 < len(W) or apply_activation_to_embedding:
                mat = activation(batchnorm(mat.dot(w), gamma, beta, moving_mean, moving_variance))
            else:
                mat = batchnorm(mat.dot(w), gamma, beta, moving_mean, moving_variance)
    if return_embedding:
        em = dataclasses.datamatrix(rowname=dm.rowname,
                                    rowlabels=dm.rowlabels.copy(),
                                    rowmeta=copy.deepcopy(dm.rowmeta),
                                    columnname='latent_component',
                                    columnlabels=np.array(['LC'+str(x) for x in range(mat.shape[1])], dtype='object'),
                                    columnmeta={},
                                    matrixname='sdae_encoding_of_'+dm.matrixname,
                                    matrix=mat.copy())
    if bn_decode_variables == None:
        for i, (w, b) in enumerate(zip(W[::-1], Bd[::-1])):
            if i+1 < len(W) or apply_activation_to_output:
                mat = activation(mat.dot(w.T) + b)
            else:
                mat = mat.dot(w.T) + b
    else:
        gammas, betas, moving_means, moving_variances = bn_decode_variables
        for i, (w, b, gamma, beta, moving_mean, moving_variance) in enumerate(zip(W[::-1], Bd[::-1], gammas, betas, moving_means, moving_variances)):
            if i+1 < len(W) or apply_activation_to_output:
                mat = activation(batchnorm(mat.dot(w.T), gamma, beta, moving_mean, moving_variance))
            else:
                mat = batchnorm(mat.dot(w.T), gamma, beta, moving_mean, moving_variance)
    rm = dataclasses.datamatrix(rowname=dm.rowname,
                                rowlabels=dm.rowlabels.copy(),
                                rowmeta=copy.deepcopy(dm.rowmeta),
                                columnname=dm.columnname,
                                columnlabels=dm.columnlabels.copy(),
                                columnmeta=copy.deepcopy(dm.columnmeta),
                                matrixname='decoding_from_sdae_encoding_of_'+dm.matrixname,
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
