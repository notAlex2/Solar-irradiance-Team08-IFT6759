import pathlib
import pandas as pd
import numpy as np
import pickle
import sys
import argparse


def filter_data(
    dfpath,
    n=0,
    stat=None
):
    '''
    dfpath: path of the data to be cleaned including the name
    n: number of images to keep before sunrise and after sunset
    stat: Station code: BND, TBL, ...If None it cleans all stations
    '''
    data = pd.read_pickle(dfpath)  # read the data
    # make sure it sorted by its index
    data.sort_index(ascending=True, inplace=True)
    data.replace('nan', np.nan, inplace=True)  # replace nan by np.nan
    # droping records without raw images
    # and hdf5 files split the dataframe by stations
    data.drop(data.loc[data.ncdf_path.isnull() & data.hdf5_8bit_path.isnull(
    )].index, inplace=True)

    BND_df = data.iloc[:, 0:9]
    TBL_df = pd.concat([data.iloc[:, 0:4], data.iloc[:, 9:13]], axis=1)
    DRA_df = pd.concat([data.iloc[:, 0:4], data.iloc[:, 13:17]], axis=1)
    FPK_df = pd.concat([data.iloc[:, 0:4], data.iloc[:, 17:21]], axis=1)
    GWN_df = pd.concat([data.iloc[:, 0:4], data.iloc[:, 21:25]], axis=1)
    PSU_df = pd.concat([data.iloc[:, 0:4], data.iloc[:, 25:29]], axis=1)
    SXF_df = pd.concat([data.iloc[:, 0:4], data.iloc[:, 29:33]], axis=1)
    Stat_df = {  # Save the station's data as a dictionnary of dataframes
        'BND': BND_df,
        'TBL': TBL_df,
        'DRA': DRA_df,
        'FPK': FPK_df,
        'GWN': GWN_df,
        'PSU': PSU_df,
        'SXF': SXF_df
    }

    print("The sequence length is :", n)
    drop_indexes = []
    if not stat:
        for df in Stat_df:
            print("Cleaning station :", df)
            data = Stat_df[df]
            DAYTIME = np.str(df + '_DAYTIME')  # Station's DAYTIME column
            GHI = np.str(df + '_GHI')  # Station's GHI column
            for j in range(0, len(data.index) - n):
                if j < n:
                    if int(data[DAYTIME][data.index[j]]) == 0 and\
                            int(data[DAYTIME][data.index[j + n]]) == 0:
                        # Storing the indexes to avoid changing the lenght
                        # of the DF while iterating
                        drop_indexes.append(j)
                else:
                    if int(data[DAYTIME][data.index[j - int(n)]]) == 0 and\
                        int(data[DAYTIME][data.index[j]]) == 0 and\
                            int(data[DAYTIME][data.index[j + int(n)]]) == 0:
                        drop_indexes.append(j)
            for j in range(len(data.index) - n, len(data.index)):
                if int(data[DAYTIME][data.index[j - n]]) == 0 and\
                        int(data[DAYTIME][data.index[j]]) == 0:
                    drop_indexes.append(j)
            data.drop(data.index[drop_indexes], inplace=True)
            print(np.str(len(drop_indexes)) +
                  " indexes dropped from station " + str(df))
            drop_indexes = []
            Stat_df[df] = pd.concat([data.iloc[:, 0:4], data[GHI]], axis=1)
        return Stat_df
    else:
        print("Cleaning station :", stat)
        data = Stat_df[stat]
        DAYTIME = np.str(stat + '_DAYTIME')
        GHI = np.str(stat + '_GHI')  # Station's GHI column
        for j in range(0, len(data.index) - int(n)):
            if j < int(n):
                if int(data[DAYTIME][data.index[j]]) == 0 and\
                        int(data[DAYTIME][data.index[j + int(n)]]) == 0:
                    # Storing the indexes to avoid changing the lenght
                    # of the DF while iterating
                    drop_indexes.append(j)
            else:
                if int(data[DAYTIME][data.index[j - int(n)]]) == 0 and\
                    int(data[DAYTIME][data.index[j]]) == 0 and\
                        int(data[DAYTIME][data.index[j + int(n)]]) == 0:
                    drop_indexes.append(j)
        for j in range(len(data.index) - n, len(data.index)):
            if int(data[DAYTIME][data.index[j - n]]) == 0 and\
                    int(data[DAYTIME][data.index[j]]) == 0:
                drop_indexes.append(j)
        data.drop(data.index[drop_indexes], inplace=True)
        print(np.str(len(drop_indexes)) +
              " indexes dropped from station " + str(stat))
        return pd.concat([data.iloc[:, 0:4], data[GHI]], axis=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input data to clean it up')
    parser.add_argument("data", type=str,
                        help="path to the data folder including its name")
    parser.add_argument("--n", type=int, default=0,
                        help="window size to keep nightime images")
    parser.add_argument("--stat", type=str, default=None,
                        help="Code of the station to be cleaned. \
        If all stations, don't put this argument")
    args = parser.parse_args()

    filter_data(
        dfpath=args.data,
        n=args.n,
        stat=args.stat
    )
