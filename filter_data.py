import pathlib
import pandas as pd
import numpy as np
import pickle

def filter_data(dfpath, n = 0, stat = None):
    ''' dfpath: path of the data to be cleaned including the name
        n: number of images to keep before sunrise and after sunset
        stat: Station code: BND, TBL, ... If None it cleans all stations'''
    
    #read the data
    data = pd.read_pickle(dfpath)
    
    #make sure it sorted by its index
    data.sort_index(ascending=True, inplace=True)
    
    #replace nan by np.nan
    data.replace('nan',np.nan, inplace=True)
    
    #droping records without raw images and hdf5 files
    data.drop(data.loc[data.ncdf_path.isnull() & data.hdf5_8bit_path.isnull()].index, inplace=True)
    
    #split the dataframe by stations
    BND_df = data.iloc[:,0:9]
    TBL_df = pd.concat([data.iloc[:,0:4],data.iloc[:,9:13]],axis=1)
    DRA_df = pd.concat([data.iloc[:,0:4],data.iloc[:,13:17]],axis=1)
    FPK_df = pd.concat([data.iloc[:,0:4],data.iloc[:,17:21]],axis=1)
    GWN_df = pd.concat([data.iloc[:,0:4],data.iloc[:,21:25]],axis=1)
    PSU_df = pd.concat([data.iloc[:,0:4],data.iloc[:,25:29]],axis=1)
    SXF_df = pd.concat([data.iloc[:,0:4],data.iloc[:,29:33]],axis=1)
    
    #Save the station's data as a dictionnary of dataframes
    Stat_df = {
        'BND':BND_df,
        'TBL':TBL_df,
        'DRA':DRA_df,
        'FPK':FPK_df,
        'GWN':GWN_df,
        'PSU':PSU_df,
        'SXF':SXF_df
    }

    print("The sequence length is :",n)
    drop_indexes = []

    if stat==None:
        for df in Stat_df:
            print("Cleaning station :",df)

            data = Stat_df[df]
            DAYTIME = np.str(df+'_DAYTIME') #Station's DAYTIME column
            
            for j in range(0,len(data.index)-n):
                if int(data[DAYTIME][data.index[j]])==0 and int(data[DAYTIME][data.index[j+n]])==0:
                    drop_indexes.append(j) #Storing the indexes to avoid changing the lenght of the DF while iterating
            
            data.drop(data.index[drop_indexes], inplace=True)
            print(np.str(len(drop_indexes))+" indexes dropped from station "+str(df))
            drop_indexes = []
            Stat_df[df]=data
    else:
        print("Cleaning station :",stat)
        data = Stat_df[stat]
        DAYTIME = np.str(stat+'_DAYTIME')
        for j in range(0,len(data.index)-n):
            if int(data[DAYTIME][data.index[j]])==0 and int(data[DAYTIME][data.index[j+n]])==0:
                drop_indexes.append(j)
        data.drop(data.index[drop_indexes], inplace=True)
        print(np.str(len(drop_indexes))+" indexes dropped from station "+str(stat))

        Stat_df[stat]=data       
    return Stat_df