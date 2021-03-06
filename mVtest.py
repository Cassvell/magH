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

################################################################################
################################################################################
################################################################################
'''
month = input('type the first three letters of the interested month: ')
pattern = '*'+month+'.22m'


def disp_coe(pattern):
    path = '/home/c-isaac/Escritorio/mag_plot/coeneo'
    filenames = next(os.walk(path))[2]
    date_list = fnmatch.filter(filenames, pattern)
#date_list = date_list.sort()
    print(date_list)

disp_coe(pattern)


def disp_aux(pattern):
    path = '/home/c-isaac/Escritorio/mag_plot/aux'
    filenames = next(os.walk(path))[2]
    date_list = fnmatch.filter(filenames, pattern)
#date_list = date_list.sort()
    print(date_list)
disp_aux(pattern)
'''
################################################################################
################################################################################
################################################################################

i_date = input("write initial date in format \n yyyy-mm-dd HH:MM:SS  " )
f_date = input("write final date in format \n yyyy-mm-dd HH:MM:SS  " )


def coe_df(i_date, f_date):
#concatenate all data into one DataFrame
#df = pd.read_csv('/home/c-isaac/Escritorio/mag_plot/coe07jan.22m', header=1, delim_whitespace=True, skip_blank_lines=True)

    #'/run/user/1001/gvfs/sftp:host=10.0.0.187/home/ccastellanos/Escritorio/proyecto/coeneo'
    #path        = '/run/user/1001/gvfs/sftp:host=132.248.208.46,user=visitante/data/magnetic_data/coeneo/2022'
    path        = '/home/c-isaac/Escritorio/mag_plot/coe'
    #name = input()
    file_names  = glob.glob(path+'/*.22v')

    dfs_c         = []

    for file_name in file_names:
        df = pd.read_csv(file_name, header=1, delim_whitespace=True, skip_blank_lines=True)
        dfs_c.append(df)
    
    df_c = pd.concat(dfs_c, axis=0, ignore_index=True)

    dd = df_c.iloc[:,0].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    mm = df_c.iloc[:,1].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    yy = df_c.iloc[:,2].astype(str)

    df_c['dt_tmp']= yy+mm+dd

#print(df)
    df_c['date']  = pd.to_datetime(df_c['dt_tmp'], format='%Y-%m-%d')    
    
#print(df['date'])
    df_c['hr']    = pd.to_timedelta(df_c.iloc[:,3], unit = 'h')
    df_c['min']   = pd.to_timedelta(df_c.iloc[:,4], unit = 'm')

    df_c['Date']  = datetimes = df_c['date'] + df_c['hr'] + df_c['min']

    dec = df_c.iloc[:,5]
    H   = df_c.iloc[:,6] 
    Z   = df_c.iloc[:,7]
    I   = df_c.iloc[:,8]
    F   = df_c.iloc[:,9]
#H = H.replace({-9999.9:np.nan})
    df_c = df_c.sort_values(by="Date")
#print(pd.isna(df))

    mask = (df_c['Date'] > i_date) & (df_c['Date'] <= f_date)
    df_c = df_c[mask]

    return (df_c)
    
################################################################################
################################################################################
################################################################################

def aux_df(i_date, f_date):
#concatenate all data into one DataFrame
#df = pd.read_csv('/home/c-isaac/Escritorio/mag_plot/coe07jan.22m', header=1, delim_whitespace=True, skip_blank_lines=True)

#'/run/user/1001/gvfs/sftp:host=10.0.0.187/home/ccastellanos/Escritorio/proyecto/coeneo'
    #path        = '/run/user/1001/gvfs/sftp:host=132.248.208.46,user=visitante/data/magnetic_data/coeneo/2022'
    path = '/home/c-isaac/Escritorio/mag_plot/aux'
#name = input()
    file_names  = glob.glob(path+'/*.22v')

    dfs_c         = []

    for file_name in file_names:
        df = pd.read_csv(file_name, header=1, delim_whitespace=True, \
        skip_blank_lines=True)
        dfs_c.append(df)
    
    df_c = pd.concat(dfs_c, axis=0, ignore_index=True)

    dd = df_c.iloc[:,0].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    mm = df_c.iloc[:,1].astype(str).apply(lambda x: '{0:0>2}'.format(x))
    yy = df_c.iloc[:,2].astype(str)

    df_c['dt_tmp']= yy+mm+dd

#print(df)
    df_c['date']  = pd.to_datetime(df_c['dt_tmp'], format='%Y-%m-%d')
#print(df['date'])
    df_c['hr']    = pd.to_timedelta(df_c.iloc[:,3], unit = 'h')
    df_c['min']   = pd.to_timedelta(df_c.iloc[:,4], unit = 'm')

    df_c['Date']  = datetimes = df_c['date'] + df_c['hr'] + df_c['min']

    dec = df_c.iloc[:,5]
    H   = df_c.iloc[:,6] 
    Z   = df_c.iloc[:,7]
    I   = df_c.iloc[:,8]
    F   = df_c.iloc[:,9]
#H = H.replace({-9999.9:np.nan})
    df_c = df_c.sort_values(by="Date")
#print(pd.isna(df))

    mask = (df_c['Date'] > i_date) & (df_c['Date'] <= f_date)
    df_c = df_c[mask]
    
    return (df_c)    

def introduce_nan(dataframe, indice): #esta sub rutina introduce los datetime faltantes, considerando los periodos en que no se registraron los datos en vez de solo pegarlos
    
    dataframe = dataframe.set_index(dataframe['Date'])
    dataframe = dataframe.reindex(indice)
    dataframe = dataframe.drop(columns=['DD', 'MM', 'YYYY', 'HH', 'MM.1', 'dt_tmp', 'hr', 'min','date', 'Date'])

    return(dataframe)

def get_jump(data):   
    #threshold = ()
    grad = pd.Series(np.gradient(data['H(mv)']), data.index, name='slope')
    q1, q3     = np.nanpercentile(grad, [5, 95])
    iqr = q3-q1
     
    idx_supjump = np.where(grad-q3 > iqr*6.0)
    idx_infjump= np.where(q1-grad > iqr*6.0)
    #print(q1, q3)

    index1=[]
    index2=[]
    #print(idx_supjump)
    for i in idx_supjump:
        #print(grad[i])
        index1.append(i)
    for j in idx_infjump:
        #print(grad[j]) 
        index2.append(j)     

    
    index1 = np.array(index1)
    index2 = np.array(index2)
    index = np.concatenate((index1, index2), axis=None)
    index = sorted(index)
    #print(index)
    #plt.plot(data.index, grad)
    #plt.show()
    return(index)
    
    
df_c  = coe_df(i_date, f_date)
df_a  = aux_df(i_date, f_date)

idx_1 = pd.date_range(start = pd.Timestamp(i_date), end = pd.Timestamp(f_date), freq='T')

dfc_wnan = introduce_nan(df_c, idx_1)
dfa_wnan = introduce_nan(df_a, idx_1)

idxc      = get_jump(dfc_wnan)
idxa      = get_jump(dfa_wnan)


dfc_wnan['H(mV)'][idxc] = np.nan
dfc_wnan['D(mV)'][idxc] = np.nan
dfc_wnan['Z(mV)'][idxc] = np.nan
dfc_wnan['Tc(mV)'][idxc] = np.nan

dfa_wnan['H(mV)'][idxa] = np.nan
dfa_wnan['D(mV)'][idxa] = np.nan
dfa_wnan['Z(mV)'][idxa] = np.nan
dfa_wnan['Tc(mV)'][idxa] = np.nan

dfc_wnan['H(mv)'][idxc] = np.nan
dfc_wnan['D(mv)'][idxc] = np.nan
dfc_wnan['Z(mv)'][idxc] = np.nan
dfc_wnan['Tc(mv)'][idxc] = np.nan

dfa_wnan['H(mv)'][idxa] = np.nan
dfa_wnan['D(mv)'][idxa] = np.nan
dfa_wnan['Z(mv)'][idxa] = np.nan
dfa_wnan['Tc(mv)'][idxa] = np.nan


inicioc = dfc_wnan.index[0]
finalc  = dfc_wnan.index[-1]


fig, ax = plt.subplots(4, figsize=(15,15))
fig.suptitle('Mediciones de Campo magn??tico de la estaci??n Coeneo \n a??o 2022', fontsize=24, fontweight='bold')
plt.subplots_adjust(hspace = .000001)

mpl.rcParams['lines.linewidth'] = 0.5

ax[0].plot(dfc_wnan.index, dfc_wnan['H(mV)'], color='g')
ax[0].plot(dfc_wnan.index, dfc_wnan['H(mv)'], color='g')
ax[0].set_ylabel('H [mV]', fontsize=18)
ax[0].set_xlim([inicioc, finalc])
ax[0].set_xticklabels(())
ax[0].grid()

ax[1].plot(dfc_wnan.index, dfc_wnan['D(mV)'], color='b')
ax[1].plot(dfc_wnan.index, dfc_wnan['D(mv)'], color='b')
ax[1].set_ylabel(' dec [mV] ', fontsize=18)
ax[1].set_xlim([inicioc, finalc])
ax[1].set_xticklabels(())
ax[1].grid()

ax[2].plot(dfc_wnan.index, dfc_wnan['Z(mV)'], color='r')
ax[2].plot(dfc_wnan.index, dfc_wnan['Z(mv)'], color='r')
ax[2].set_ylabel('Z [mV]', fontsize=18)
ax[2].set_xlim([inicioc, finalc])
ax[2].set_xticklabels(())
ax[2].grid()

ax[3].plot(dfc_wnan.index, dfc_wnan['Tc(mV)'], color='k')
ax[3].plot(dfc_wnan.index, dfc_wnan['Tc(mv)'], color='k')
ax[3].set_ylabel('Tc [mV}', fontsize=18)
ax[3].set_xlim([inicioc, finalc])
ax[3].set_xlabel('Time series', fontsize=18)
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%B-%d %H:%M"))
ax[3].grid()

#plt.show()
pathfig = '/home/c-isaac/Escritorio/mag_plot/figures/'
namefig = 'coe_mv'+i_date+' to '+f_date
plt.savefig(pathfig+namefig+'.png')



fig, ax = plt.subplots(4, figsize=(15,15))
fig.suptitle('Mediciones de Campo magn??tico de la estaci??n Auxiliar en Coeneo \n a??o 2022', \
             fontsize=24, fontweight='bold')
plt.subplots_adjust(hspace = .000001)

mpl.rcParams['lines.linewidth'] = 0.5

ax[0].plot(dfa_wnan.index, dfa_wnan['H(mV)'], color='g')
ax[0].plot(dfa_wnan.index, dfa_wnan['H(mv)'], color='g')
ax[0].set_ylabel('H [mV]', fontsize=18)
ax[0].set_xlim([inicioc, finalc])
ax[0].set_xticklabels(())
ax[0].grid()

ax[1].plot(dfa_wnan.index, dfa_wnan['D(mV)'], color='b')
ax[1].plot(dfa_wnan.index, dfa_wnan['D(mv)'], color='b')
ax[1].set_ylabel(' dec [mV] ', fontsize=18)
ax[1].set_xlim([inicioc, finalc])
ax[1].set_xticklabels(())
ax[1].grid()

ax[2].plot(dfa_wnan.index, dfa_wnan['Z(mV)'], color='r')
ax[2].plot(dfa_wnan.index, dfa_wnan['Z(mv)'], color='r')
ax[2].set_ylabel('Z [mV]', fontsize=18)
ax[2].set_xlim([inicioc, finalc])
ax[2].set_xticklabels(())
ax[2].grid()

ax[3].plot(dfa_wnan.index, dfa_wnan['Tc(mV)'], color='k')
ax[3].plot(dfa_wnan.index, dfa_wnan['Tc(mv)'], color='k')
ax[3].set_ylabel('Tc(mV)', fontsize=18)
ax[3].set_xlim([inicioc, finalc])
ax[3].set_xlabel('Time series', fontsize=18)
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%B-%d %H:%M"))
ax[3].grid()

#plt.show()
namefig2 = 'aux_mv'+i_date+' to '+f_date
plt.savefig(pathfig+namefig2+'.png')

#img = Image.open('/home/c-isaac/Im??genes/119-1190732_warning-icon-png-transparent-background-warning-icon-png.jpeg')
#img.show()
fig, ax = plt.subplots(4, figsize=(15,15))
fig.suptitle('Mediciones de Campo magn??tico de la estaci??n Auxiliar en Coeneo \n a??o 2022', \
             fontsize=24, fontweight='bold')
plt.subplots_adjust(hspace = .000001)

mpl.rcParams['lines.linewidth'] = 0.5

ax[0].plot(dfa_wnan.index, dfa_wnan['H(mV)'], 'b', label='aux')
ax[0].plot(dfc_wnan.index, dfc_wnan['H(mV)'], 'k', label='coe')
ax[0].plot(dfa_wnan.index, dfa_wnan['H(mv)'], 'b', label='aux')
ax[0].plot(dfc_wnan.index, dfc_wnan['H(mv)'], 'k', label='coe')
ax[0].set_ylabel('H [mv]', fontsize=18)
ax[0].set_xlim([inicioc, finalc])
ax[0].set_xticklabels(())
ax[0].grid()
ax[0].legend()

ax[1].plot(dfa_wnan.index, dfa_wnan['D(mV)'], 'b', label='aux')
ax[1].plot(dfc_wnan.index, dfc_wnan['D(mV)'], 'k', label='coe')
ax[1].plot(dfa_wnan.index, dfa_wnan['D(mv)'], 'b', label='aux')
ax[1].plot(dfc_wnan.index, dfc_wnan['D(mv)'], 'k', label='coe')
ax[1].set_ylabel(' dec [mV] ', fontsize=18)
ax[1].set_xlim([inicioc, finalc])
ax[1].set_xticklabels(())
ax[1].grid()
ax[1].legend()

ax[2].plot(dfa_wnan.index, dfa_wnan['Z(mV)'], 'b', label='aux')
ax[2].plot(dfc_wnan.index, dfc_wnan['Z(mV)'], 'k', label='coe')
ax[2].plot(dfa_wnan.index, dfa_wnan['Z(mv)'], 'b', label='aux')
ax[2].plot(dfc_wnan.index, dfc_wnan['Z(mv)'], 'k', label='coe')
ax[2].set_ylabel('Z [mV]', fontsize=18)
ax[2].set_xlim([inicioc, finalc])
ax[2].set_xticklabels(())
ax[2].grid()
ax[2].legend()

ax[3].plot(dfa_wnan.index, dfa_wnan['Tc(mV)'], 'b', label='aux')
ax[3].plot(dfc_wnan.index, dfc_wnan['Tc(mV)'], 'k', label='coe')
ax[3].plot(dfa_wnan.index, dfa_wnan['Tc(mv)'], 'b', label='aux')
ax[3].plot(dfc_wnan.index, dfc_wnan['Tc(mv)'], 'k', label='coe')
ax[3].set_ylabel('Tc [mV])', fontsize=18)
ax[3].set_xlim([inicioc, finalc])
ax[3].set_xlabel('Time series', fontsize=18)
ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%B-%d %H:%M"))
ax[3].grid()
ax[3].legend()
#plt.show()
namefig2 = 'comp_mv'+i_date+' to '+f_date
plt.savefig(pathfig+namefig2+'.png')

