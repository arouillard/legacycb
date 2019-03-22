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


def create_and_save_partitions(dataset, study_name, test_fraction=0.1, valid_fraction=0.1, save_text_files=False):
    
    # determine dataset orientation
    orientation = 'skinny' if dataset.shape[0] > dataset.shape[1] else 'fat'
    
    # partition the data
    tobepopped = np.random.permutation(dataset.shape[0]) < round(max([test_fraction*dataset.shape[0], 2.0]))
    dataset_test = dataset.pop(tobepopped, 0)
    print('    TEST', flush=True)
    print(dataset_test)
    tobepopped = np.random.permutation(dataset.shape[0]) < round(max([valid_fraction*dataset.shape[0], 2.0]))
    dataset_valid = dataset.pop(tobepopped, 0)
    print('    VALID', flush=True)
    print(dataset_valid)
    dataset_train = dataset
    print('    TRAIN', flush=True)
    print(dataset_train)
    
    # save data partitions
    savefolder = '../partitioned_data/{0}/{1}'.format(study_name, orientation)
    print('    SAVING PARTITIONS TO {0}'.format(savefolder), flush=True)
    os.makedirs(savefolder)
    datasetIO.save_datamatrix('{0}/test.pickle'.format(savefolder), dataset_test)
    datasetIO.save_datamatrix('{0}/valid.pickle'.format(savefolder), dataset_valid)
    datasetIO.save_datamatrix('{0}/train.pickle'.format(savefolder), dataset_train)
    if save_text_files:
        datasetIO.save_datamatrix('{0}/test.txt.gz'.format(savefolder), dataset_test)
        datasetIO.save_datamatrix('{0}/valid.txt.gz'.format(savefolder), dataset_valid)
        datasetIO.save_datamatrix('{0}/train.txt.gz'.format(savefolder), dataset_train)


def main(study_name, row_data_path=None, column_data_path=None, matrix_data_path=None, partition_axis=-1, dtype='float64', delimiter='\t', test_fraction=0.1, valid_fraction=0.1, save_text_files=True):

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
    print(dataset)
    
    # shuffle the data
    dataset.reorder(np.random.permutation(dataset.shape[0]), 0)
    dataset.reorder(np.random.permutation(dataset.shape[1]), 1)
    
    # partition the data
    if partition_axis == 1 or partition_axis == -1:
        print('PARTITIONING TRANSPOSE...', flush=True)
        create_and_save_partitions(dataset.totranspose(), study_name, test_fraction, valid_fraction, save_text_files)
    if partition_axis == 0 or partition_axis == -1:
        print('PARTITIONING ORIGINAL...', flush=True)
        create_and_save_partitions(dataset, study_name, test_fraction, valid_fraction, save_text_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Partition dataset into training, validation, and test sets')
    parser.add_argument('study_name', help='name (and subfolder) for your study', type=str)
    parser.add_argument('--row_data_path', help='path to row data', type=str)
    parser.add_argument('--column_data_path', help='path to column data', type=str)
    parser.add_argument('--matrix_data_path', help='path to matrix data', type=str)
    parser.add_argument('--partition_axis', help='axis to partition (0=rows, 1=columns, -1=both)', type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument('--dtype', help='data type of matrix', type=str, default='float64', choices=['bool', 'int32', 'int64', 'float32', 'float64'])
    parser.add_argument('--delimiter', help='delimiter of input files', type=str, default='\t')
    parser.add_argument('--test_fraction', help='fraction of data to reserve for model testing', type=float, default=0.1)
    parser.add_argument('--valid_fraction', help='fraction of data to reserve for model validation', type=float, default=0.1)
    parser.add_argument('--save_text_files', help='save partitioned datasets as text files in addition to .pickle files (otherwise only save .pickle files)', type=bool, default=True, choices=[False, True])
    args = parser.parse_args()    
    main(args.study_name, args.row_data_path, args.column_data_path, args.matrix_data_path, args.partition_axis, args.dtype, args.delimiter, args.test_fraction, args.valid_fraction, args.save_text_files)

