# classify storm by sub-basins and adjust the storm durations based on their presence in the sub-basin
# @author: Yuan Liu
# 2021/12/21


import os
import numpy as np
import pandas as pd
import xarray as xr
from initial_storm_catalog import make_folder
from time import time


def storm_basin_classify(year, area_threshold, prcp_threshold, save_loc):
    """
    Classify storms by sub-basins and adjust durations. If the storm area or precipitation over the sub-basin meet the
    threshold, the storm is classified to the sub-basin and the duration of the storm is adjusted to the duration when
    the storm is in the sub-basin.
    :param year: the year of the processed storm catalog
    :param area_threshold: area threshold to classify the storm to one sub-basin, default is 2500 km^2
    :param prcp_threshold: precipitation threshold to classify the storm to one sub-basin, default is 0.1 mm/hour
    :param save_loc: location to save sub-basin storm catalog
    :return:
    """
    # record the start time
    ts = time()
    print("Classify sub-basin storm in {0}".format(year))
    make_folder(save_loc)
    # load sbasin boundary array, where the grid inside the sub-basin is 1, otherwise is 0
    low_miss_array = np.load("boundary_files/low_miss_basin_array_boundary.npy")
    arkansas_array = np.load("boundary_files/arkansas_basin_array_boundary.npy")
    missouri_array = np.load("boundary_files/missouri_basin_array_boundary.npy")
    ohio_array = np.load("boundary_files/ohio_basin_array_boundary.npy")
    up_miss_array = np.load("boundary_files/up_miss_basin_array_boundary.npy")
    miss_array = np.load("boundary_files/miss_basin_array_boundary.npy")

    # load hourly storm record table processed by initial_storm_catalog.py
    storm_catalog = pd.read_pickle(os.path.join(save_loc, "land_storm_catalog_" + str(year) + ".pkl"))

    # initialize sub-basin storm catalog dataframe
    miss_catalog = pd.DataFrame()
    arkansas_catalog = pd.DataFrame()
    missouri_catalog = pd.DataFrame()
    ohio_catalog = pd.DataFrame()
    up_miss_catalog = pd.DataFrame()
    low_miss_catalog = pd.DataFrame()
    # the full catalog record storms in all the sub-basins
    full_storm_catalog = pd.DataFrame()

    # create a dictionary to save the catalogs
    sbasin_df_dict = {"miss": miss_catalog, "arkansas": arkansas_catalog,
                      "missouri": missouri_catalog, "ohio": ohio_catalog,
                      "up_miss": up_miss_catalog, "low_miss": low_miss_catalog}

    sbasin_name_list = ['miss', 'arkansas', 'low_miss', 'missouri', 'up_miss', 'ohio']
    # create a dictionary to save the sub-basin boundary
    sbasin_array_dict = {'miss': miss_array, 'arkansas': arkansas_array, 'low_miss': low_miss_array,
                         'missouri': missouri_array, 'up_miss': up_miss_array, 'ohio': ohio_array}
    # get storm ids from catalog
    storm_ids = storm_catalog['ID'].unique()
    for i in range(storm_ids.shape[0]):
        storm_id = storm_ids[i]
        # load the storm precipitation spatial pattern
        single_record_folder = os.path.join(save_loc, "single_record")
        storm_record = xr.load_dataset(os.path.join(single_record_folder, storm_id + ".nc"))
        # extract the storm record from storm catalog
        storm_record_stats = storm_catalog[storm_catalog['ID'] == storm_id]
        # reset the index of the storm record
        storm_record_stats = storm_record_stats.reset_index(drop=True)
        # get the total time steps of the storm record
        lifetime = storm_record_stats.shape[0]
        # read precipitation field
        land_mtpr_array = storm_record['land_mtpr'].data
        # read the longitude and latitude coordinates
        lon_data = storm_record['lon'].data
        lat_data = storm_record['lat'].data
        grid_cell_degree = 0.25
        lon_2, lat_2 = np.meshgrid(lon_data, lat_data)
        # compute the projected area of each pixel
        pixel_area = np.cos(lat_2 * np.pi / 180) * 111 * 111 * grid_cell_degree * grid_cell_degree

        sbasin_indicator_dict = {'miss': [], 'arkansas': [], 'low_miss': [], 'missouri': [], 'up_miss': [], 'ohio': []}
        sbasin_avg_mtpr_dict = {'miss': [], 'arkansas': [], 'low_miss': [], 'missouri': [], 'up_miss': [], 'ohio': []}
        sbasin_avg_area_dict = {'miss': [], 'arkansas': [], 'low_miss': [], 'missouri': [], 'up_miss': [], 'ohio': []}
        land_cen_lon_list = []
        land_cen_lat_list = []

        for time_index in range(lifetime):
            # compute the centroid of the land part of the storm
            land_mtpr_field = land_mtpr_array[time_index]
            sum_precipitation = np.sum(land_mtpr_field)
            x_center = np.sum(np.where(land_mtpr_field, ((lon_2 * land_mtpr_field) /
                                                         sum_precipitation), 0))
            y_center = np.sum(np.where(land_mtpr_field, ((lat_2 * land_mtpr_field) /
                                                         sum_precipitation), 0))
            # append to centroid list
            land_cen_lon_list.append(x_center)
            land_cen_lat_list.append(y_center)

            for sbasin_name in sbasin_name_list:
                # extract the boundary of current sub-basin
                sbasin_array = sbasin_array_dict[sbasin_name]
                # extract the overlap precipitation area between the sub-basin and current storm
                sbasin_overlap_mtpr = np.where(sbasin_array == 1, land_mtpr_field, 0)
                # compute the overlapping area
                sbasin_overlap_area = pixel_area[sbasin_overlap_mtpr != 0].sum()
                # compute the area-weighted average precipitation
                if sbasin_overlap_area != 0:
                    sbasin_avg_mtpr = ((sbasin_overlap_mtpr * pixel_area).sum()) / sbasin_overlap_area
                else:
                    sbasin_avg_mtpr = 0
                # if the overlapping area is larger than the area threshold, or the precipitation is larger
                # than the precipitation threshold, the storm is considered to be in the sub-basin.
                if (sbasin_overlap_area >= area_threshold) | (sbasin_avg_mtpr >= prcp_threshold):
                    sbasin_indicator = 1
                else:
                    sbasin_indicator = 0

                sbasin_avg_mtpr_dict[sbasin_name].append(sbasin_avg_mtpr)
                sbasin_avg_area_dict[sbasin_name].append(sbasin_overlap_area)
                sbasin_indicator_dict[sbasin_name].append(sbasin_indicator)
        # record the centroid of the land part of the storm
        storm_record_stats['land_cen_lat'] = land_cen_lat_list
        storm_record_stats['land_cen_lon'] = land_cen_lon_list
        # append list to dataframe
        for sbasin_name in sbasin_name_list:
            storm_record_stats[sbasin_name + "_area_avg_mtpr(mm/h)"] = sbasin_avg_mtpr_dict[sbasin_name]
            storm_record_stats[sbasin_name + "_area(sqkm)"] = sbasin_avg_area_dict[sbasin_name]
            storm_record_stats[sbasin_name + '_storm'] = sbasin_indicator_dict[sbasin_name]
        # record the storm in the full storm catalog
        full_storm_catalog = full_storm_catalog.append(storm_record_stats, ignore_index=True)
        # adjust the storm duration based on its presence in sub-basins
        for sbasin_name in sbasin_name_list:
            # deep copy the storm record
            sbasin_land_storm_record = storm_record_stats.copy(deep=True)
            sbasin_land_storm_record = sbasin_land_storm_record.reset_index(drop=True)
            record_length = sbasin_land_storm_record.shape[0]
            start_index = 0
            for i in range(record_length):
                # if it is zero for the time step
                if sbasin_land_storm_record.loc[i, sbasin_name + "_storm"] == 0:
                    # print('Remove sea storm record')
                    sbasin_land_storm_record.drop(i, axis=0, inplace=True)
                    start_index = start_index + 1
                else:
                    break
            if sbasin_land_storm_record.shape[0] == 0:
                # print('The storm is empty.')
                continue
            end_index = record_length
            for i in range(record_length):
                if sbasin_land_storm_record.loc[record_length - 1 - i, sbasin_name + "_storm"] == 0:
                    # print('Remove sea storm record')
                    sbasin_land_storm_record.drop(record_length - 1 - i, axis=0, inplace=True)
                    end_index = end_index - 1
                else:
                    break
            if sbasin_land_storm_record.shape[0] == 0:
                print('Storm not in {0}'.format(sbasin_name))
                continue
            # save the new storm durations
            storm_duration = sbasin_land_storm_record.shape[0]
            storm_day = round(storm_duration / 24)
            storm_duration_list = [storm_duration] * storm_duration
            storm_day_list = [storm_day] * storm_duration

            sbasin_land_storm_record['sbasin_duration(hour)'] = storm_duration_list
            sbasin_land_storm_record['sbasin_durationday(day)'] = storm_day_list

            # append the record to the sbasin storm catalog
            sbasin_df_dict[sbasin_name] = sbasin_df_dict[sbasin_name].append(sbasin_land_storm_record,
                                                                             ignore_index=True)
    # save the dataframe
    for sbasin_name in sbasin_name_list:
        sbasin_df_dict[sbasin_name].to_pickle(
            os.path.join(save_loc, sbasin_name + "_land_storm_catalog_" + str(year) + ".pkl"))
    print("Finish classify sub-basin storm in {0} in time {1}s".format(year, time() - ts))


if __name__ == "__main__":
    year = 2019
    catalog_loc = "hourly_catalog"
    area_threshold = 2500
    prcp_threshold = 0.1
    storm_basin_classify(year, area_threshold, prcp_threshold, save_loc=catalog_loc)