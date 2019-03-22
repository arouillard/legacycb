# -*- coding: utf-8 -*-
"""
@author: ar988996
"""


import sys
sys.path.append('../utilities')

import numpy as np
import datasetIO
import os
import argparse


def create_and_save_partitions(dataset, study_name, meta_label, test_groups, pretest_groups, valid_groups, save_text_files=True):
    
    # determine dataset orientation
    orientation = 'skinny' if dataset.shape[0] > dataset.shape[1] else 'fat'
    
    # discard null categories
    tobediscarded = np.in1d(dataset.rowmeta[meta_label], ['-666', '', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'NAN', 'nan'])
    dataset.discard(tobediscarded, 0)
    print('discarding {0!s} samples...'.format(tobediscarded.sum()), flush=True)
    print(dataset, flush=True)
    
    # partition the data
    tobepopped = np.in1d(dataset.rowmeta[meta_label], test_groups)
    dataset_test = dataset.pop(tobepopped, 0)
    print('    TEST', flush=True)
    print(dataset_test, flush=True)
    tobepopped = np.in1d(dataset.rowmeta[meta_label], pretest_groups)
    dataset_pretest = dataset.pop(tobepopped, 0)
    print('    PRETEST', flush=True)
    print(dataset_pretest, flush=True)
    tobepopped = np.in1d(dataset.rowmeta[meta_label], valid_groups)
    dataset_valid = dataset.pop(tobepopped, 0)
    print('    VALID', flush=True)
    print(dataset_valid, flush=True)
    dataset_train = dataset
    print('    TRAIN', flush=True)
    print(dataset_train, flush=True)
    
    # save data partitions
    savefolder = '../partitioned_data/{0}/{1}'.format(study_name, orientation)
    print('    SAVING PARTITIONS TO {0}'.format(savefolder), flush=True)
    os.makedirs(savefolder)
    datasetIO.save_datamatrix('{0}/test.pickle'.format(savefolder), dataset_test)
    datasetIO.save_datamatrix('{0}/pretest.pickle'.format(savefolder), dataset_pretest)
    datasetIO.save_datamatrix('{0}/valid.pickle'.format(savefolder), dataset_valid)
    datasetIO.save_datamatrix('{0}/train.pickle'.format(savefolder), dataset_train)
    if save_text_files:
        os.mkdir('{0}/test'.format(savefolder))
        datasetIO.save_splitdata('{0}/test'.format(savefolder), dataset_test)
        os.mkdir('{0}/pretest'.format(savefolder))
        datasetIO.save_splitdata('{0}/pretest'.format(savefolder), dataset_pretest)
        os.mkdir('{0}/valid'.format(savefolder))
        datasetIO.save_splitdata('{0}/valid'.format(savefolder), dataset_valid)
        os.mkdir('{0}/train'.format(savefolder))
        datasetIO.save_splitdata('{0}/train'.format(savefolder), dataset_train)
#        datasetIO.save_datamatrix('{0}/test.txt.gz'.format(savefolder), dataset_test)
#        datasetIO.save_datamatrix('{0}/pretest.txt.gz'.format(savefolder), dataset_pretest)
#        datasetIO.save_datamatrix('{0}/valid.txt.gz'.format(savefolder), dataset_valid)
#        datasetIO.save_datamatrix('{0}/train.txt.gz'.format(savefolder), dataset_train)


def main(study_name, meta_label, test_groups, pretest_groups, valid_groups, row_data_path=None, column_data_path=None, matrix_data_path=None, partition_axis=0, dtype='float64', delimiter='\t', save_text_files=True):

    print('study_name: {0}'.format(study_name), flush=True)
    print('meta_label: {0}'.format(meta_label), flush=True)
    print('test_groups:', test_groups, flush=True)
    print('pretest_groups:', pretest_groups, flush=True)
    print('valid_groups:', valid_groups, flush=True)
    
    # load data and create datamatrix object
    if row_data_path==None or column_data_path==None or matrix_data_path==None:
        loadfolder = '../input_data/{0}'.format(study_name)
        if os.path.exists(loadfolder):
            original_files = os.listdir(loadfolder)
            rowhit = ['rowdata.txt' in x for x in original_files]
            columnhit = ['columndata.txt' in x for x in original_files]
            matrixhit = ['matrixdata.txt' in x for x in original_files]
            if sum(rowhit) > 0 and sum(columnhit) > 0 and sum(matrixhit) > 0:
                row_data_path = '{0}/{1}'.format(loadfolder, original_files[rowhit.index(True)])
                column_data_path = '{0}/{1}'.format(loadfolder, original_files[columnhit.index(True)])
                matrix_data_path = '{0}/{1}'.format(loadfolder, original_files[matrixhit.index(True)])
                print('LOADING DATA...', flush=True)
                print('    row_data_path: {0}'.format(row_data_path), flush=True)
                print('    column_data_path: {0}'.format(column_data_path), flush=True)
                print('    matrix_data_path: {0}'.format(matrix_data_path), flush=True)
                dataset = datasetIO.load_splitdata(row_data_path, column_data_path, matrix_data_path, study_name, dtype, delimiter)
            else:
                hit = ['datamatrix' in x for x in original_files]
                if sum(hit) > 0:
                    datamatrix_path = '{0}/{1}'.format(loadfolder, original_files[hit.index(True)])
                    print('LOADING DATA...', flush=True)
                    print('    datamatrix_path: {0}'.format(datamatrix_path), flush=True)
                    dataset = datasetIO.load_datamatrix(datamatrix_path)
                else:
                    raise ValueError('input data incorrectly specified')
        else:
            raise ValueError('input data incorrectly specified')
    else:
        print('LOADING DATA...', flush=True)
        print('    row_data_path: {0}'.format(row_data_path), flush=True)
        print('    column_data_path: {0}'.format(column_data_path), flush=True)
        print('    matrix_data_path: {0}'.format(matrix_data_path), flush=True)
        dataset = datasetIO.load_splitdata(row_data_path, column_data_path, matrix_data_path, study_name, dtype, delimiter)
    print('ORIGINAL', flush=True)
    print(dataset, flush=True)
    
    # shuffle the data
    dataset.reorder(np.random.permutation(dataset.shape[0]), 0)
    dataset.reorder(np.random.permutation(dataset.shape[1]), 1)
    
    # partition the data
    if partition_axis == 1:
        print('PARTITIONING TRANSPOSE...', flush=True)
        create_and_save_partitions(dataset.totranspose(), study_name, meta_label, test_groups, pretest_groups, valid_groups, save_text_files)
    elif partition_axis == 0:
        print('PARTITIONING ORIGINAL...', flush=True)
        create_and_save_partitions(dataset, study_name, meta_label, test_groups, pretest_groups, valid_groups, save_text_files)
    else:
        raise ValueError('invalid partition_axis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Partition dataset into training, validation, and test sets')
    parser.add_argument('study_name', help='name (and subfolder) for your study', type=str)
    parser.add_argument('--meta_label', help='label of metadata variable used to select holdout groups', type=str)
    parser.add_argument('--test_groups', help='groups of samples to reserve for model testing', type=str, nargs='+')
    parser.add_argument('--pretest_groups', help='groups of samples to reserve for model pretesting', type=str, nargs='+')
    parser.add_argument('--valid_groups', help='groups of samples to reserve for model validation', type=str, nargs='+')
    parser.add_argument('--row_data_path', help='path to row data', type=str)
    parser.add_argument('--column_data_path', help='path to column data', type=str)
    parser.add_argument('--matrix_data_path', help='path to matrix data', type=str)
    parser.add_argument('--partition_axis', help='axis to partition (0=rows, 1=columns)', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dtype', help='data type of matrix', type=str, default='float64', choices=['bool', 'int32', 'int64', 'float32', 'float64'])
    parser.add_argument('--delimiter', help='delimiter of input files', type=str, default='\t')
    parser.add_argument('--save_text_files', help='save partitioned datasets as text files in addition to .pickle files (otherwise only save .pickle files)', type=bool, default=True, choices=[False, True])
    args = parser.parse_args()    
    main(args.study_name, args.meta_label, args.test_groups, args.pretest_groups, args.valid_groups, args.row_data_path, args.column_data_path, args.matrix_data_path, args.partition_axis, args.dtype, args.delimiter, args.save_text_files)

