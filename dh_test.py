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

import itertools

from scipy import signal
################################################################################
################################################################################
################################################################################
'''
month = input('type the first three letters of the interested month: ')
pattern = '*'+month+'.22m'


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
################################################################################
################################################################################
################################################################################
def coe_df(i_date, f_date):
#concatenate all data into one DataFrame
#df = pd.read_csv('/home/c-isaac/Escritorio/mag_plot/coe07jan.22m', header=1, delim_whitespace=True, skip_blank_lines=True)

#'/run/user/1001/gvfs/sftp:host=10.0.0.187/home/ccastellanos/Escritorio/proyecto/coeneo'
    #path        = '/run/user/1001/gvfs/sftp:host=132.248.208.46,user=visitante/data/magnetic_data/coeneo/2022'
    path = '/home/c-isaac/Escritorio/mag_plot/coe'
#name = input()
    file_names  = glob.glob(path+'/*.22m')

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

    Date = df_c['Date'][mask]
    H            = H[mask]
    dec          = dec[mask]
    Z            = Z[mask]
    I            = I[mask]
    F            = F[mask]
 
    return (df_c)
################################################################################
################################################################################
################################################################################            
'''
def plot_rawdata(date, h):

    fig = plt.figure(figsize=(15,15))
    #fig.suptitle('Mediciones de Campo magnético de la estación Auxiliar en Coeneo \n año 2022', \
     #            fontsize=24, fontweight='bold')
    plt.plot(date, h, linewidth=1.0, color='k')
    plt.xlabel('Time UT')
    plt.ylabel('H [nT]')
    plt.grid()
    plt.tight_layout()
    #plt.xlim([0, 1400])
    
    pathfig     = '/home/c-isaac/Escritorio/mag_plot/raw_plots/'
    namefig2 = 'coe'+i_date+' to '+f_date
    plt.savefig(pathfig+namefig2+'.png')



plot_raw = plot_rawdata(df['Date'], df['H(nT)'])
'''
################################################################################
################################################################################
################################################################################
df = coe_df(i_date, f_date)  
#df = df.set_index(date)
################################################################################
#Detrend Time series
################################################################################
df['H(nT)'] = signal.detrend(df['H(nT)'])
#df = df.reset_index(drop=True, inplace=True)

idx_1 = pd.date_range(start = pd.Timestamp(i_date), end = pd.Timestamp(f_date), freq='T')
#print(df)    

################################################################################
################################################################################
################################################################################
def introduce_nan(dataframe, indice): #esta sub rutina introduce los datetime faltantes, considerando los periodos en que no se registraron los datos en vez de solo pegarlos
    
    dataframe = dataframe.set_index(dataframe['Date'])
    dataframe = dataframe.reindex(indice)
    dataframe = dataframe.drop(columns=['DD', 'MM', 'YYYY', 'HH', 'MM.1', 'dt_tmp', 'hr', 'min','date', 'Date'])

    return(dataframe)
################################################################################
################################################################################
################################################################################

df_wnan = introduce_nan(df, idx_1) #dataframe que considera el tiempo en que no se registraron datos

'''
def plot_rawdata(date, h):

    fig = plt.figure(figsize=(15,15))
    #fig.suptitle('Mediciones de Campo magnético de la estación Auxiliar en Coeneo \n año 2022', \
     #            fontsize=24, fontweight='bold')
    plt.plot(date, h, linewidth=1.0, color='k')
    plt.xlabel('Time UT')
    plt.ylabel('H [nT]')
    plt.grid()
    plt.tight_layout()
    #plt.xlim([0, 1400])
    
    pathfig     = '/home/c-isaac/Escritorio/mag_plot/raw_plots/'
    namefig2 = 'coe_withnan'+i_date+' to '+f_date
    plt.savefig(pathfig+namefig2+'.png')
plot_wnan = plot_rawdata(df_wnan.index df_wnan['H(nT)'])
'''
'''
def remove_spikes(dataframe): #quitar los picos y artefactos causados por el magnetometro
    
    q1, q3     = np.percentile(dataframe, [25, 75])
    iqr        = q3-q1
    dataframe[((dataframe - q3) > 1.5*iqr) & ((q1 - dataframe) > 1.5*iqr)] = np.nan
    dataframe['H(nT)'].plot()
    plt.show
    return(dataframe)
'''
################################################################################
################################################################################
################################################################################
'''
def outliers_iqrfilter(data)
    q1, q3     = np.percentile(data, [25, 75])
    iqr        = q3-q1
    
    #data[((data - q3) > 1.5*iqr) & ((data) > 1.5*iqr)] = np.nan
    #print(iqr)
    return(ndata)
'''
################################################################################
################################################################################
################################################################################
def get_jump(data):   
    #threshold = ()
    grad = pd.Series(np.gradient(data['H(nT)']), data.index, name='slope')
    q1, q3     = np.nanpercentile(grad, [5, 95])
    iqr = q3-q1
     
    idx_supjump = np.where(grad-q3 > iqr*6)
    idx_infjump= np.where(q1-grad > iqr*6)
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

jump = get_jump(df_wnan) #Eliminamos los saltos en los datos que pudieran deberse a algún ruido del instrumento
H    = df_wnan['H(nT)']
H[jump] = np.nan

#imprimir PWS de componente H

def B_longtime(data):
    medhr = data.resample('60min').median()
    length = len(medhr)
    for i in range(length):   
        print(medhr[i*24:(i+1)*24-1])
    
    #idx   = time + pd.Timedelta(30, unit='m')
    #medhr = medhr.set_index(idx)

    #plt.plot(medhr.index, medhr['H(nT)'])
    #plt.show()
#
  # medd = medhr.resample('D').median()
 #   idx_d = medd.index + pd.Timedelta(12, unit='h')
#    medd = medd.reindex(idx_d)
    
    #i_medd = medd.interpolate(limit=12, limit_direction = 'both')
   # i_medd = i_medd.rolling('15D').mean()
    
  #  re_ixmed = i_medd.reindex(data.index)
    
 #   Bt = re_ixmed.interpolate(limit=7220, limit_direction='both')

  #  return(Bt)
################################################################################
Bt = B_longtime(df_wnan)
#H    = Bt['H(nT)']
#plt.plot(df_wnan.index, Bt)
#plt.show()
#medianas cada hora
#print(H[jump])
#print(H[jump])
#print(df_wnan['H(nT)'][jump])
#print(df_wnan['H(nT)'][jump])
'''
path_dst = '/home/c-isaac/Escritorio/mag_plot/dst/'
file_name2 = path_dst+'ASY_'+i_date+'_'+f_date+'m_P.dat'
df_dst = pd.read_csv(file_name2, header=24, sep='\s+', skip_blank_lines=True)


sym = df_dst['SYM-H'] 

inicioc = df_wnan.index[0]
finalc  = df_wnan.index[-1]

plt.plot(df_wnan.index, H, linewidth=1.0, label='experimental DH')
plt.plot(df_wnan.index, sym, 'k', linewidth=1.0, label='SYM-H')
plt.xlim([inicioc, finalc])
plt.grid()
plt.legend()
plt.show()

'''





