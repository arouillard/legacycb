# -*- coding: utf-8 -*-
"""
@author: ar988996
"""

#import sys
#custompaths = ['/GWD/bioinfo/projects/cb01/users/rouillard/Python/Classes',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Modules',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Packages',
#               '/GWD/bioinfo/projects/cb01/users/rouillard/Python/Scripts']
##custompaths = ['C:\\Users\\ar988996\\Documents\\Python\\Classes',
##               'C:\\Users\\ar988996\\Documents\\Python\\Modules',
##               'C:\\Users\\ar988996\\Documents\\Python\\Packages',
##               'C:\\Users\\ar988996\\Documents\\Python\\Scripts']
#for custompath in custompaths:
#    if custompath not in sys.path:
#        sys.path.append(custompath)
#del custompath, custompaths
#
#from machinelearning import datasetIO
#
#p = 'data/prepared_data/GTEXv6old/skinny/'
#for f in ['train.pickle', 'test.pickle', 'valid.pickle']:
#    print(f, flush=True)
#    d = datasetIO.load_datamatrix(p + f)
#    datasetIO.save_datamatrix(p + f.replace('.pickle', '.txt.gz'), d)

import datasetIO

p = 'data/prepared_data/GTEXv6old/skinny/'
for f in ['train.pickle', 'test.pickle', 'valid.pickle']:
    print(f, flush=True)
    d = datasetIO.load_datamatrix(p + f.replace('pickle', 'txt.gz'))
    datasetIO.save_datamatrix(p + f, d)
