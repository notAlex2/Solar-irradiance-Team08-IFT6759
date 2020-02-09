import pathlib
import pandas as pd
import numpy as np
import pickle
import sys
import argparse
from model_logging import get_logger

logging = get_logger()


def clean_one_station(Stat_df, station_id, drop_indexes, nb_images):
    logging.debug("Cleaning station :{0}".format(station_id))
    data = Stat_df[station_id]
    DAYTIME = np.str(station_id + '_DAYTIME')
    GHI = np.str(station_id + '_GHI')  # Station's GHI column

    for j in range(len(data.index) - int(nb_images)):
        if j < int(nb_images):
            if int(data[DAYTIME][data.index[j]]) == 0 and\
               int(data[DAYTIME][data.index[j + int(nb_images)]]) == 0:
                # Storing the indexes to avoid changing the lenght
                # of the DF while iterating
                drop_indexes.append(j)
        else:
            if int(data[DAYTIME][data.index[j - int(nb_images)]]) == 0 and\
               int(data[DAYTIME][data.index[j]]) == 0 and\
               int(data[DAYTIME][data.index[j + int(nb_images)]]) == 0:
                drop_indexes.append(j)

    for j in range(len(data.index) - nb_images, len(data.index)):
        if int(data[DAYTIME][data.index[j - nb_images]]) == 0 and\
           int(data[DAYTIME][data.index[j]]) == 0:
            drop_indexes.append(j)

    data.drop(data.index[drop_indexes], inplace=True)
    logging.debug(np.str(len(drop_indexes)) + " indexes dropped from station " + str(station_id))
    return pd.concat([data.iloc[:, 0:4], data[GHI]], axis=1)


def filter_data(
    df_path,
    nb_images=0,
    station_id=None
):
    '''
    df_path: path of the data to be cleaned including the name
    nb_images: number of images to keep before sunrise and after sunset
    station_id: Station code: BND, TBL, ...If None it cleans all stations
    '''
    data = pd.read_pickle(df_path)  # read the data

    # make sure it sorted by its index
    data.sort_index(ascending=True, inplace=True)

    # replace nan by np.nan
    data.replace('nan', np.nan, inplace=True)

    # droping records without raw images AND hdf5 files
    data.drop(data.loc[data.ncdf_path.isnull() & data.hdf5_8bit_path.isnull()].index, inplace=True)

    # split the dataframe by stations
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

    logging.debug("The sequence length is :{0}".format(nb_images))
    if station_id:
        # Clean one station
        drop_indexes = []
        return {station_id: clean_one_station(Stat_df, station_id, drop_indexes, nb_images)}
    else:
        # Clean all stations
        for df in Stat_df:
            drop_indexes = []
            Stat_df[df] = clean_one_station(Stat_df, df, drop_indexes, nb_images)
        return Stat_df


def parse_args():
    parser = argparse.ArgumentParser(description='input data to clean it up')
    parser.add_argument("data", type=str, help="path to the data folder including its name")
    parser.add_argument("--n", type=int, default=0, help="window size to keep nightime images")
    parser.add_argument("--stat", type=str, default=None, help="Code of the station to be cleaned. \
        If all stations, don't put this argument")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data = filter_data(
        df_path=args.data,
        nb_images=args.n,
        station_id=args.stat
    )

    for station_id, df in data.items():
        df.to_csv("log/filtered_{0}_df.csv".format(station_id))
