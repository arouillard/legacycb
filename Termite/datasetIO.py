# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""

import os
import gzip
import copy
import pickle
import numpy as np
import dataclasses as dc

def load_datamatrix(datasetpath, delimiter='\t', dtype='float64', getmetadata=True, getmatrix=True):
    if '.pickle' in datasetpath:
        with open(datasetpath, 'rb') as fr:
            return pickle.load(fr)
    else:
        if '.gz' in datasetpath:
            openfunc = gzip.open
        else:
            openfunc = open
        with openfunc(datasetpath, mode='rt', encoding="utf-8", errors="surrogateescape") as fr:
            rowmeta = {}
            columnmeta = {}
            rowlabels = []
            entries = [x.strip() for x in fr.readline().split(delimiter)]
            skipcolumns = sum([entry=='#' for entry in entries]) + 1
            columnname = entries[skipcolumns-1]
            columnlabels = np.array(entries[skipcolumns:], dtype='object')
            firstentry = entries[0]
            skiprows = 1
            if getmetadata:
                while firstentry == '#':
                    entries = [x.strip() for x in fr.readline().split(delimiter)]
                    columnmetaname = entries[skipcolumns-1].split('/')[-1]
                    if columnmetaname.lower() != 'na':
                        columnmeta[columnmetaname] = np.array(entries[skipcolumns:], dtype='object')
                    firstentry = entries[0]
                    skiprows += 1
                rowname = firstentry
                rowmetanames = entries[1:skipcolumns]
                if len(rowmetanames) > 0:
                    rowmetanames[-1] = rowmetanames[-1].split('/')[0]
                rowmetaname_idx = {}
                for i, rowmetaname in enumerate(rowmetanames):
                    if rowmetaname.lower() != 'na':
                        rowmeta[rowmetaname] = []
                        rowmetaname_idx[rowmetaname] = i
                for line in fr:
                    entries = [x.strip() for x in line.split(delimiter, maxsplit=skipcolumns)[:skipcolumns]]
                    rowlabels.append(entries.pop(0))
                    for rowmetaname, idx in rowmetaname_idx.items():
                        rowmeta[rowmetaname].append(entries[idx])
                rowlabels = np.array(rowlabels, dtype='object')
                for rowmetaname, rowmetavalues in rowmeta.items():
                    rowmeta[rowmetaname] = np.array(rowmetavalues, dtype='object')
            else:
                while firstentry == '#':
                    entries = [x.strip() for x in fr.readline().split(delimiter)]
                    firstentry = entries[0]
                    skiprows += 1
                rowname = firstentry
                for line in fr:
                    rowlabels.append(line.split(delimiter, maxsplit=1)[0].strip())
                rowlabels = np.array(rowlabels, dtype='object')
        if getmatrix:
            matrix = np.loadtxt(datasetpath, dtype=dtype, delimiter=delimiter, skiprows=skiprows,
                                usecols=range(skipcolumns,len(columnlabels)+skipcolumns), ndmin=2)
        else:
            matrix = np.zeros((0,0), dtype=dtype)
        matrixname = rowname + '_' + columnname + '_associations_from_' + datasetpath
        return dc.datamatrix(rowname, rowlabels, columnname, columnlabels, matrixname, matrix, rowmeta, columnmeta)

def save_datamatrix(datasetpath, dm):
    if '.pickle' in datasetpath:
        with open(datasetpath, 'wb') as fw:
            pickle.dump(dm, fw)
    else:
        if '.gz' in datasetpath:
            openfunc = gzip.open
        else:
            openfunc = open
        np.savetxt(datasetpath.replace('.txt', '.temp.txt'), dm.matrix, fmt='%1.6g', delimiter='\t', newline='\n')
        with openfunc(datasetpath, mode='wt', encoding="utf-8", errors="surrogateescape") as fw, openfunc(datasetpath.replace('.txt', '.temp.txt'), 'rt') as fr:
            rowmeta_names_and_dtypes = []
            for rowmetaname, rowmetadata in dm.rowmeta.items():
                if len(rowmetadata.shape) > 1:
                    dm.rowmeta[rowmetaname] = rowmetadata.reshape(-1)
                rowmeta_names_and_dtypes.append((rowmetaname, rowmetadata.dtype))
            spacers = ['#' for x in range(len(rowmeta_names_and_dtypes)+1)]
            if dm.columnlabels.dtype == 'object':
                fw.write('\t'.join(spacers + [dm.columnname] + dm.columnlabels.tolist()) + '\n')
            else:
                fw.write('\t'.join(spacers + [dm.columnname] + ['{0:1.6g}'.format(x) for x in dm.columnlabels]) + '\n')
            for columnmetaname, columnmetadata in dm.columnmeta.items():
                if len(columnmetadata.shape) > 1:
                    columnmetadata = columnmetadata.reshape(-1)
                if columnmetadata.dtype == 'object':
                    fw.write('\t'.join(spacers + [columnmetaname] + columnmetadata.tolist()) + '\n')
                else:
                    fw.write('\t'.join(spacers + [columnmetaname] + ['{0:1.6g}'.format(x) for x in columnmetadata]) + '\n')
            fw.write('\t'.join([dm.rowname] + [k for k,t in rowmeta_names_and_dtypes] + ['na/na'] + ['na' for i in range(dm.shape[1])]) + '\n')
            for i, line in enumerate(fr):
                rowmetadata = [dm.rowmeta[k][i] if t=='object' else '{0:1.6g}'.format(dm.rowmeta[k][i]) for k,t in rowmeta_names_and_dtypes]
                fw.write('\t'.join([dm.rowlabels[i]] + rowmetadata + ['na']) + '\t' + line)
        os.remove(datasetpath.replace('.txt', '.temp.txt'))

def load_metadata(metadatapath, delimiter='\t'):
    if '.gz' in metadatapath:
        openfunc = gzip.open
    else:
        openfunc = open
    with openfunc(metadatapath, mode='rt', encoding='utf-8', errors='surrogateescape') as fr:
        fields = [x.strip() for x in fr.readline().split(delimiter)]
        field_data = {field:np.empty(0, dtype='object') for field in fields}
        for line in fr:
            values = [x.strip() for x in line.split(delimiter)]
            for i, field in enumerate(fields):
                field_data[field] = np.insert(field_data[field], field_data[field].size, values[i])
    if field_data[fields[0]].size > np.unique(field_data[fields[0]]).size:
        raise ValueError('labels in {0} are not unique'.format(metadatapath))
    return fields[0], field_data[fields[0]].copy(), field_data

def load_splitdata(rowdatapath, columndatapath, matrixdatapath, studyname='', dtype='float64', delimiter='\t', matrix_has_labels=True):
    rowname, rowlabels, rowmeta = load_metadata(rowdatapath, delimiter)
    columnname, columnlabels, columnmeta = load_metadata(columndatapath, delimiter)
    if matrix_has_labels:
        matrix = np.loadtxt(matrixdatapath, dtype=dtype, delimiter=delimiter, skiprows=1, usecols=range(1,len(columnlabels)+1), ndmin=2)
    else:
        matrix = np.loadtxt(matrixdatapath, dtype=dtype, delimiter=delimiter, ndmin=2)
    if studyname == '':
        studyname = matrixdatapath
    matrixname = '{0}-{1}_data_from_{2}'.format(rowname, columnname, studyname)
    return dc.datamatrix(rowname, rowlabels, columnname, columnlabels, matrixname, matrix, rowmeta, columnmeta)

def save_metadata(metadatapath, fields, field_data, delimiter='\t'):
    if '.gz' in metadatapath:
        openfunc = gzip.open
    else:
        openfunc = open
    with openfunc(metadatapath, mode='wt', encoding='utf-8', errors='surrogateescape') as fw:
        fw.write(delimiter.join(fields) + '\n')
        for i in range(field_data[fields[0]].size):
            fw.write(delimiter.join([field_data[f][i] if field_data[f].dtype=='object' else '{0:1.6g}'.format(field_data[f][i]) for f in fields]) + '\n')

def save_splitdata(folderpath, dm, delimiter='\t', compress=True, add_labels_to_matrixdata=True):
    extension = 'csv' if delimiter == ',' else 'txt'
    extension += '.gz' if compress else ''
    fields = [dm.rowname] + list(dm.rowmeta.keys())
    field_data = copy.deepcopy(dm.rowmeta)
    field_data[dm.rowname] = dm.rowlabels
    save_metadata('{0}/rowdata.{1}'.format(folderpath, extension), fields, field_data, delimiter)
    fields = [dm.columnname] + list(dm.columnmeta.keys())
    field_data = copy.deepcopy(dm.columnmeta)
    field_data[dm.columnname] = dm.columnlabels
    save_metadata('{0}/columndata.{1}'.format(folderpath, extension), fields, field_data, delimiter)
    if add_labels_to_matrixdata:
        np.savetxt('{0}/matrixdata.temp.{1}'.format(folderpath, extension), dm.matrix, fmt='%1.6g', delimiter=delimiter, newline='\n')
        openfunc = gzip.open if compress else open
        with openfunc('{0}/matrixdata.{1}'.format(folderpath, extension), mode='wt', encoding="utf-8", errors="surrogateescape") as fw, openfunc('{0}/matrixdata.temp.{1}'.format(folderpath, extension), 'rt') as fr:
            fw.write(delimiter.join([dm.rowname] + dm.columnlabels.tolist()) + '\n')
            for i, line in enumerate(fr):
                fw.write(dm.rowlabels[i] + delimiter + line)
        os.remove('{0}/matrixdata.temp.{1}'.format(folderpath, extension))
    else:
        np.savetxt('{0}/matrixdata.{1}'.format(folderpath, extension), dm.matrix, fmt='%1.6g', delimiter=delimiter, newline='\n')
