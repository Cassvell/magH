import glob, os
import fnmatch 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

from PIL import Image
from datetime import datetime, timedelta
import glob, os
import fnmatch 

from os import listdir
from os.path import isfile, join

path        = '/home/c-isaac/Escritorio/mag_plot/coeneo'
#name = input()
file_names  = glob.glob(path+'/*feb.22m')

dfs_c         = []

for file_name in file_names:
        df = pd.read_csv(file_name, index_col=None, header=1, delim_whitespace=True)
        dfs_c.append(df)
    
df_c = pd.concat(dfs_c, axis=0, ignore_index=True)
#df_c.dropna(df_c, axis=1,  how="any")
print(df_c)


#month = input('type the first three letters of the interested month: ')
#pattern = '*'+month+'*'


#def disp_coe(pattern):
#    path = '/run/user/1001/gvfs/sftp:host=132.248.208.46,user=visitante/data\
#    /magnetic_data/coeneo/2022/DataMin'
#    filenames = next(os.walk(path))[2]
#    date_list = fnmatch.filter(filenames, pattern)
#date_list = date_list.sort()
#    print(date_list)

#disp_coe(pattern)


#def disp_aux(pattern):
#    path = '/run/user/1001/gvfs/sftp:host=132.248.208.46,\
#    user=visitante/data/magnetic_data/auxiliar/2022/DataMin'
#    filenames = next(os.walk(path))[2]
#    date_list = fnmatch.filter(filenames, pattern)
#date_list = date_list.sort()
#    print(date_list)
#disp_aux(pattern)




