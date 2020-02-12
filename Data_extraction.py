import pathlib
import pandas as pd
import numpy as np
import pickle
import sys
import argparse
import json
import typing
import logging
import h5py
import utils
import tensorflow as tf

# from model_logging import get_logger

# logging = get_logger()


def clean_one_station(Stat_df, station_id, drop_indexes, nb_images):
    # logging.debug("Cleaning station :{0}".format(station_id))
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
    # logging.debug(np.str(len(drop_indexes)) +
    # " indexes dropped from station " + str(station_id))
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

    # logging.debug("The sequence length is :{0}".format(nb_images))
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

#############################################
# Extract data from HDF5 files
#############################################


def get_stations_coordinates(
    hdf5_path: str,
    stations_lats_lons: typing.Dict[str, typing.Tuple]
) -> typing.Dict[str, typing.Tuple]:
    """
    :param datafram_path: str pointing to the dataframe .pkl file
    :param stations_lats_lons: dictionnary of str -> (latitude, longitude) of the station(s)
    :return: dictionnary of str -> (coord_x, coord_y) in the numpy array
    """
    # Give instead the HDF5 file's path
    # df = pd.read_pickle(dataframe_path) if dataframe_path else None
    # takes the first non Nan or 'nan' value for the hdf5 path
    # hdf5_path = df.loc[df["hdf5_8bit_path"].notnull() == True & (df["hdf5_8bit_path"] != 'nan')]["hdf5_8bit_path"][0]

    with h5py.File(hdf5_path, 'r') as h5_data:
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, 0), utils.fetch_hdf5_sample("lon", h5_data, 0)
    stations_coords = {}
    for region, lats_lons in stations_lats_lons.items():
        coords = (np.argmin(np.abs(lats - lats_lons[0])), np.argmin(np.abs(lons - lats_lons[1])))
        stations_coords[region] = coords
    return stations_coords

# need to modify this so that the normalization takes the min, max of each channel and not each image


def normalize_images(
    image: np.ndarray
):
    """
    :param image: image as an array
    :return: an array that is normalized between 0 and 1
    """
    image = (image - np.min(image)) / np.ptp(image)
    return image


def generate_images(
    cropped_image: np.ndarray,
    station: str,
    file_date: str,
    offset: int
):
    """
    :param cropped_image: array of a cropped station
    :param station: string -> station that was cropped
    :param file_day: str -> corresponding date of the file that is cropped
    :param offset: the integer index (or offset) that corresponds to the position of the sample in the dataset
    This function will save the cropped images as .h5 file in the format of cropped_filedate_station_offset.h5
    """
    # Can specify the path here at the beginning of the argument. Note that w stands for create file, truncate if exists
    with h5py.File(f"cropped_{file_date}" + "_" + f"{station}" + "_" + f"{offset}.hdf5", "w") as f:
        crop = f.create_dataset("images", data=cropped_image)
        # can add other keys if needed


def save_img(img, ch, offset, pathname):
    img = np.around(img*255/(img.max()-img.min()))
    img = cv2.flip(img, 0)
    name = "./"+np.str(pathname)+"."+np.str(offset)+"."+np.str("ch1")+".png"
    # print(name)
    cv2.imwrite(name, img)
    return name


def crop_images(
    hdf5_path: str,
    window_size: float,
    img_compression: str,
    hdf5_offset: int,
    stations: typing.Dict[str, typing.Tuple]
):
    """
    :param datafram_path: str pointing to the dataframe .pkl file
    :param window_size : float defining the pixel range of the crop centered at a station. One pixel is 16km^2
    :return:
    """
    assert window_size < 42, f"window_size value of {window_size} is too big, please reduce it to 42 and lower"

    # Already done in filter_data
    # df = pd.read_pickle(dataframe_path) if dataframe_path else None
    # df_copy = df.copy().replace(to_replace="nan",
    #                            value=np.NaN).dropna(subset=["hdf5_8bit_path"])

    # Construting the HDF5 column name depending on the compression type 8bit or 16bit
    HDF5_Path_col = np.str("hdf5_"+img_compression+"_path")
    HDF5_Offset_col = np.str("hdf5_" + img_compression + "_offset")

    img_crop = {}

    with h5py.File(np.str(hdf5_path), "r") as h5_data:  # it is not efficient to open and close H5 file 96 times for 1 day!

        # Get the stations pixel coordonates from lats and lons
        # coordinates = get_stations_coordinates(h5_data, extract_config["stations"])
        lats, lons = utils.fetch_hdf5_sample("lat", h5_data, 0), utils.fetch_hdf5_sample("lon", h5_data, 0)
        stations_coords = {}
        for region, lats_lons in stations.items():
            coords = (np.argmin(np.abs(lats - lats_lons[0])), np.argmin(np.abs(lons - lats_lons[1])))
            stations_coords[region] = coords

        # print(stations_coords)

        # normalize arrays and crop the station for each channel
        # ML: I'm not sure if we need to normalize at this level
        ch1_data = normalize_images(utils.fetch_hdf5_sample("ch1", h5_data, hdf5_offset))
        ch2_data = normalize_images(utils.fetch_hdf5_sample("ch2", h5_data, hdf5_offset))
        ch3_data = normalize_images(utils.fetch_hdf5_sample("ch3", h5_data, hdf5_offset))
        ch4_data = normalize_images(utils.fetch_hdf5_sample("ch4", h5_data, hdf5_offset))
        ch6_data = normalize_images(utils.fetch_hdf5_sample("ch6", h5_data, hdf5_offset))

        for station_coordinates in stations_coords.items():
            # retrieves station name and coordinates for each station
            station_name = station_coordinates[0]
            x_coord = station_coordinates[1][0]
            y_coord = station_coordinates[1][1]

            ch1_crop = ch1_data[x_coord - window_size:x_coord + window_size, y_coord - window_size:y_coord + window_size]
            # save_img(ch1_crop, "ch1", hdf5_offset, hdf5_path)
            ch2_crop = ch2_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
            ch3_crop = ch3_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
            ch4_crop = ch4_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]
            ch6_crop = ch6_data[x_coord - window_size:x_coord + window_size, y_coord-window_size:y_coord + window_size]

            img_crop.update({hdf5_offset: np.stack((ch1_crop, ch2_crop, ch3_crop, ch4_crop, ch6_crop), axis=-1)})
            # save the images as .h5 file, will need to specify path
            # generate_images(img_crop, station_name, file_date, hdf5_offset)

        # dataset = tf.data.Dataset.from_tensor_slices((HDF5_Offset_col, img_crop))  # add DF index, GHI, CS_GHI

        return img_crop


if __name__ == '__main__':

    # Read the parameters
    print("Read the parameters ...")
    with open('pre_process_cfg_local.json', 'r') as fd:
        extract_config = json.load(fd)

    # Read and filter DataFrame
    print("Read and clean the Dataframe ...")
    clean_data = filter_data(extract_config["dataframe_path"],
                             extract_config["sequence_len"],
                             'BND')

    # We don't need to loop over DF rows as H5 file is daily and contains all 15min records (96 records if there are no missing data)
    # We need instead loop over the batch_size which can be a couple of days, each day has one H5 file.
    # for index, row in dataframe.iterrows():

    # print("clean_data.keys()=", clean_data.keys())
    record_days = {}
    h5_files_path = {}
    batch = extract_config["batch_size"]
    HDF5_Path_col = np.str("hdf5_" + extract_config["image_compression"] + "_path")
    HDF5_Offset_col = np.str("hdf5_" + extract_config["image_compression"] + "_offset")

    dataset = {}

    for station in clean_data.keys():
        print("Nbr of stations: ", len(clean_data.keys()))
        # record_days.update({station: np.unique(clean_data[station].index.strftime("%Y-%m-%d"))})
        for i in range(0, len(clean_data[station]) // batch, batch):
            print("Nbr of batchs: ", len(clean_data[station]) // batch)
            # clean_data[station].index.isin(pd.to_datetime(record_days[i:i+batch]))][HDF5_Path_col].unique()
            h5_files_path = clean_data[station].iloc[i:i + batch, :][HDF5_Path_col].unique()

            print("Nbr of H5 files to process: ", len(h5_files_path))
            print("List of h5 files to process=", h5_files_path)

            for j in range(0, len(h5_files_path)):
                hdf5_path = extract_config["hdf5_path"]  # h5_files_path[j] because its running localy
                print("Processing h5 file(s):", hdf5_path)
                hdf5_offset = 32  # to be corrected, loop over all day offset :clean_data[station].loc[clean_data[station][HDF5_Path_col] == hdf5_path][HDF5_Offset_col]  # 32
                print("hdf5_offset = ", hdf5_offset)
                # Extract and fetch and crop images from HDF5 files
                print("Extract and crop channels from HDF5 files ...")
                img_cropped = crop_images(hdf5_path, extract_config["crop_shape"], extract_config["image_compression"], hdf5_offset, extract_config["stations"])
                dataset.update({station: img_cropped})
                break
            break
        break
    for i in dataset.keys():
        print("dataset size = ", np.str(i), len(dataset[i]))
        print("Shape of the cropped imgs: ", dataset[i][32].shape)
    # file_date = hdf5_path.split("/")[-1]
    # date of the file
    # file_date = "_".join(file_date.split('.'))[:-3]
    # hdf5_offset = 32  # row[HDF5_Offset_col]
