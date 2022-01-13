# Area duration selector algorithm to find storms with specific area and duration
# @author: Yuan Liu
# 2021/01/10

import os
import numpy as np
import pandas as pd
import xarray as xr
import skimage.morphology
from time import time
import warnings
from initial_storm_catalog import make_folder


def return_neighbor_index(index, number_neighbor, boundary_array):
    """
    Return the index of neighboring pixels
    :param index: the index (row, col) of the current pixel
    :param number_neighbor: neighbor range, default 1 pixel
    :param boundary_array: the limit of the array boundary
    :return: return indices of neighboring pixels
    """
    left = max(0, index[1] - number_neighbor)
    right = min(boundary_array.shape[1] - 1, index[1] + number_neighbor)

    bottom = min(boundary_array.shape[0] - 1, index[0] + number_neighbor)
    top = max(0, index[0] - number_neighbor)

    candidate_pixel = np.array([[top, index[1]], [bottom, index[1]], [index[0], left], [index[0], right]])
    # find and remove candidate pixel with raw index
    candidate_pixel = candidate_pixel[np.sum(candidate_pixel == index, axis=1) != 2]
    # return unique labels
    return np.unique(candidate_pixel, axis=0)


def greedy_area_searcher(area_of_interest, land_prcp_array, pixel_area):
    """
    Greedy algorithm to find the largest precipitation region with specific area
    :param area_of_interest: the area of interest (sqkm)
    :param land_prcp_array: storm precipitation region 
    :param pixel_area: array that contains area (sqkm) for each pixel
    :return:
    kept_area: area of the selected precipitation region
    kept_area_prcp: area-weighted precipitation of the selected precipitation region
    kept_prcp_array: precipitation field of the selected region
    kept_prcp_binary: 0-1 binary field of the selected region with 1 means in the region, 0 otherwise
    """
    # Use binary search to find the contour of precipitation that is closest to the area of interest
    # get the largest storm precipitation
    high_bound = land_prcp_array.max()
    # get the lowest storm precipitation
    low_bound = land_prcp_array.min()
    # set the current precipitation threshold as the mean value between the highest and lowest precipitation
    curr_threshold = (high_bound - low_bound) / 2
    # initialize the selected storm area
    kept_area = 9999999
    selected_area_list = []
    selected_error_list = []
    selected_avg_prcp_list = []
    selected_prcp_field_list = []

    # first, we use the binary search to find the contour of precipitation that has the highest precipitation and
    # closest area to the area of interest (aoi)
    while abs(area_of_interest - kept_area) > 100:
        # if the difference between aoi and selected area < 100 sqkm, then stops the binary search
        # find the precipitation region above the current threshold
        boundary_loc = land_prcp_array > curr_threshold
        # set the grid with precipitation to 1, otherwise 0
        curr_land_prcp_binary = np.where(boundary_loc, 1, 0)
        # label contiguous regions as storm centers
        connect_label = skimage.morphology.label(curr_land_prcp_binary, connectivity=2)
        # for each storm center, compute the average precipitation, and find the region with the highest precipitation
        center_ids = np.unique(connect_label)
        kept_area_prcp = 0
        kept_area = 0
        kept_prcp_array = np.ones(curr_land_prcp_binary.shape)
        for center_id in center_ids:
            # skip the background
            if center_id == 0:
                continue
            # get the storm center extend
            center_extent = (connect_label == center_id)
            # compute the extent area
            center_area = np.sum(np.where(center_extent, pixel_area, 0))
            # get the precipitation field
            center_prcp_array = np.where(center_extent, land_prcp_array, 0)
            # compute the area-weighted average precipitation
            center_avg_ncg = (center_prcp_array * pixel_area).sum() / center_area
            # find the center with the largest area-weighted average precipitation
            if center_avg_ncg > kept_area_prcp:
                kept_area = center_area
                kept_area_prcp = center_avg_ncg
                kept_prcp_array = center_prcp_array
        # compute the difference between the selected contour area and the area of interest
        error = area_of_interest - kept_area
        # record the selected area, error, average intensity, and precipitation field
        selected_area_list.append(kept_area)
        selected_error_list.append(error)
        selected_avg_prcp_list.append(kept_area_prcp)
        selected_prcp_field_list.append(kept_prcp_array)

        # update the threshold based on result at the previous step
        # if the aoi > kept area, reduce the threshold to the mean of the lowest and previous thresholds
        # if the aoi < kept area, rise the threshold to the mean of the previous and highest thresholds
        lower_quantile = (curr_threshold - low_bound) / 2 + low_bound
        higher_quantile = (high_bound - curr_threshold) / 2 + curr_threshold
        if area_of_interest >= kept_area:
            high_bound = curr_threshold
            curr_threshold = lower_quantile
        else:
            low_bound = curr_threshold
            curr_threshold = higher_quantile

        # if the search lasting longer than 4 round, and the area converges to the same number, then stop the search
        if len(selected_area_list) >= 4:
            if (selected_area_list[-1] == selected_area_list[-2]) & (selected_area_list[-2] == selected_area_list[-3]):
                break

    # find the region that has the smallest error while smaller than the AOI
    selected_error_list = np.array(selected_error_list)
    error = selected_error_list[selected_error_list > -600].min()
    index = np.argwhere(selected_error_list == error)[0][0]
    kept_area = selected_area_list[index]
    kept_area_prcp = selected_avg_prcp_list[index]
    kept_prcp_array = selected_prcp_field_list[index]

    kept_prcp_binary = np.where(kept_prcp_array != 0, 1, 0)
    # compute the difference between the AOI and the kept region
    area_diff = area_of_interest - kept_area
    # if AOI is higher than the kept region by over one pixel
    if area_diff > 400:
        kept_prcp_binary = np.where(kept_prcp_array != 0, 1, 0)
        # dilate the kept region twice
        dilated_binary_array = skimage.morphology.dilation(kept_prcp_binary)
        dilated_binary_array = skimage.morphology.dilation(dilated_binary_array)
        # get the boundary of the dilated regions
        boundary_binary_array = (kept_prcp_binary == 0) & (dilated_binary_array != 0)
        # find the precipitation data in that boundary
        boundary_prcp_array = np.where(boundary_binary_array, land_prcp_array, 0)
        # find precipitation grid that is larger than 0
        non_zero_boundary_prcp = boundary_prcp_array[boundary_prcp_array != 0]
        # find the coordinate indices of the precipitation grid
        non_zero_boundary_coordinate = np.argwhere(boundary_prcp_array != 0)
        # compute pixel area array for the boundary region
        # this array and list is the candidate array of neighboring precipitation pixels that to be expanded
        non_zero_pixel_area_array = np.where(boundary_prcp_array != 0, pixel_area, 0)
        non_zero_bixel_area_list = non_zero_pixel_area_array[non_zero_pixel_area_array != 0]

        # greedy algorithm to expand the kept region with neighboring pixel that has the highest precipitation
        # create a copy as the history array of precipitation region
        history_record_array = dilated_binary_array
        while non_zero_boundary_coordinate.shape[0] > 0:
            # find the maximum prcp pixel in the candidate array
            current_max = non_zero_boundary_prcp.max()
            # find the maximum prcp index in the candidate array
            current_max_index = np.argwhere(non_zero_boundary_prcp == current_max)[0][0]
            # find the coord, and area of that maximum pixel
            current_max_coord = non_zero_boundary_coordinate[current_max_index]
            current_max_pixel_area = non_zero_bixel_area_list[current_max_index]
            # add this pixel in the kept area
            x_max = current_max_coord[0]
            y_max = current_max_coord[1]
            kept_prcp_binary[x_max, y_max] = 1
            # add the area of this pixel in dialate area
            kept_area = kept_area + current_max_pixel_area
            # remove the maximum record from the candidate array
            non_zero_boundary_prcp = np.delete(non_zero_boundary_prcp, current_max_index)
            non_zero_bixel_area_list = np.delete(non_zero_bixel_area_list, current_max_index)
            non_zero_boundary_coordinate = np.delete(non_zero_boundary_coordinate, current_max_index, axis=0)

            # find the neighboring pixels of the maximum pixel and add them to the candidate array
            number_neighbor = 1
            candidate_pixel = return_neighbor_index(current_max_coord, number_neighbor, boundary_prcp_array)
            for i in range(candidate_pixel.shape[0]):
                x = candidate_pixel[i, 0]
                y = candidate_pixel[i, 1]

                if history_record_array[x, y] == 1:
                    # this pixel is already in the dilated precipitation area and does not need to be added
                    continue
                else:
                    # record the added pixel in history array
                    history_record_array[x, y] = 1
                    # append the precipitation data in the candidate array
                    non_zero_boundary_prcp = np.append(non_zero_boundary_prcp, land_prcp_array[x, y])
                    # append the pixel area in the candidate array
                    non_zero_bixel_area_list = np.append(non_zero_bixel_area_list, pixel_area[x, y])
                    # append the coordinate in the candidate array
                    non_zero_boundary_coordinate = np.append(non_zero_boundary_coordinate, candidate_pixel[[i]], axis=0)

            # compute the difference of expanded region and AOI
            area_diff = area_of_interest - kept_area
            if abs(area_diff) < 400:
                # if the difference is smaller than one pixel, stop the algorithm
                # print("Area difference < 400 sqkm, converge")
                break
        # compute the final selected area
        kept_area = np.sum(np.where(kept_prcp_binary != 0, pixel_area, 0))
        # extract the final precipitation field
        kept_prcp_array = np.where(kept_prcp_binary != 0, land_prcp_array, 0)
        # compute the final average precipitation
        kept_area_prcp = (kept_prcp_array * pixel_area).sum() / kept_area
        # print("New filled area is {0} with avg prcp {1} mm\n".format(kept_area, kept_area_prcp))
    # else:
    # print("Selected area is {0} with avg prcp {1} mm\n".format(kept_area, kept_area_prcp))

    return kept_area, kept_area_prcp, kept_prcp_array, kept_prcp_binary


def area_duration_selector(area_of_interest, duration_of_interest, storm_stats, storm_xarray, sbasin_name,
                           sbasin_array, xarray_save_Loc, era_field):
    """
    Area duration selector that search the largest precipitation event with specific duration and area
    :param area_of_interest: target area of the region for the storm event
    :param duration_of_interest: target duration of the storm event
    :param storm_stats: dataframe of single storm record
    :param storm_xarray: xarray of the storm record
    :param sbasin_name: sub-basin name
    :param sbasin_array: su-bbasin boundary array
    :param era_field: dictionary of the moisture variable fields from the ERA5 data
    :return: dataframe record of the selected storm event
    """
    # a temporal dataframe to store the storm record
    temp_dataframe = pd.DataFrame()
    # get the pixel area
    lon_data = storm_xarray['lon'].data
    lat_data = storm_xarray['lat'].data
    # compute the area
    grid_cell_degree = 0.25
    lon_2, lat_2 = np.meshgrid(lon_data, lat_data)
    pixel_area = np.cos(lat_2 * np.pi / 180) * 111 * 111 * grid_cell_degree * grid_cell_degree
    # reset the index of the dataframe of the storm record
    storm_stats = storm_stats.reset_index(drop=False)
    # compute the average precipitation within the time window of the target duration
    rolling_mean = storm_stats.rolling(duration_of_interest).mean()
    # sort the rolling mean
    sorted_rolling_mean = rolling_mean[sbasin_name + "area_avg_mtpr(mm/h)"].sort_values(ascending=False)
    # drop nan
    sorted_rolling_mean.dropna(axis=0, how='any', inplace=True)

    for i in range(sorted_rolling_mean.shape[0]):
        # starting from the time window with the highest precipitation
        # get the current time index
        curr_index = sorted_rolling_mean.index.values[i]
        # get the timestamp range
        timestamp_range = storm_stats.loc[
            np.arange(curr_index - duration_of_interest + 1, curr_index + 1), 'Timestamp'].values
        start_time = timestamp_range[0]
        end_time = timestamp_range[-1]
        # get the xarray of the current time window
        select_daily_storm_nc = storm_xarray.sel(time=timestamp_range)
        # average the precipitation field across the time
        mean_select_daily_storm_nc = select_daily_storm_nc.mean(dim="time")
        # get the precipitation data array
        land_mtpr_array = mean_select_daily_storm_nc['land_mtpr'].data
        # only keep the precipitation inside the sub-basin
        land_mtpr_array = np.where(sbasin_array==1, land_mtpr_array, 0)
        # run area searcher
        # select the region with area of interest
        kept_area, kept_avg_mtpr, kept_mtpr_array, kept_mtpr_binary = greedy_area_searcher(area_of_interest,
                                                                                           land_mtpr_array, pixel_area)
        # compute the intensity weighted centroid of selected avg ncg
        sum_precipitation = np.sum(kept_mtpr_array)
        # if the search result returns a region with zero precipitation, continue with the next time window
        if sum_precipitation == 0:
            strong_sbasin_storm = 0
            continue
        # compute the storm centroid
        x_avg = np.sum(np.where(kept_mtpr_array, ((lon_2 * kept_mtpr_array) /
                                                  sum_precipitation), 0))
        y_avg = np.sum(np.where(kept_mtpr_array, ((lat_2 * kept_mtpr_array) /
                                                  sum_precipitation), 0))
        # check if the centroid locates in the sub-basin
        round_lon = round(x_avg * 4) / 4
        round_lat = round(y_avg * 4) / 4
        try:
            row_index = np.argwhere(lat_data == round_lat)[0][0]
            col_index = np.argwhere(lon_data == round_lon)[0][0]
        except:
            # if the centroid is not in the center, try next time window
            continue
        # if the center is inside the sub-basin
        if sbasin_array[row_index, col_index] == 1:
            strong_sbasin_storm = 1
            break
        else:
            strong_sbasin_storm = 0
        # if the algorithm fails to find the event with target duration and area
        if i == sorted_rolling_mean.shape[0] - 1:
            strong_sbasin_storm = 0

    # append storm information to new temp dataframe
    storm_id = storm_stats['ID'].values[0]
    raw_duration_hour = storm_stats['sbasin_duration(hour)'].values[0]
    raw_timestamp = storm_stats['Timestamp'].values[0]
    season = storm_stats['Season'].values[0]
    season_id = storm_stats['Season_id'].values[0]
    month = storm_stats['Month'].values[0]

    # get the average distance and bearing in the time window
    raw_avg_distance = rolling_mean.loc[curr_index, 'Distance(km)'] # average speed during the duration
    raw_avg_bearing = rolling_mean.loc[curr_index, 'Bearing(degree)'] # average bearing during the duration
    raw_avg_land_area = rolling_mean.loc[curr_index, 'Land_area(sqkm)'] # average area duration the duration

    # indicator for if the storm center in the sbasin
    sbasin_binary = strong_sbasin_storm
    temp_dataframe['ID'] = [storm_id]
    temp_dataframe['start_time'] = [start_time]
    temp_dataframe['end_time'] = [end_time]
    temp_dataframe['subset_cen_lon'] = [x_avg] # update new center location
    temp_dataframe['subset_cen_lat'] = [y_avg] # update new center location
    temp_dataframe['subset_duration(hour)'] = [duration_of_interest]
    temp_dataframe['subset_area(sqkm)'] = [area_of_interest]
    temp_dataframe['subset_ture_area(sqkm)'] = [kept_area]
    temp_dataframe['raw_duration(hour)'] = [raw_duration_hour]
    temp_dataframe['raw_timestamp(hour)'] = [raw_timestamp]
    temp_dataframe['season'] = [season]
    temp_dataframe['season_id'] = [season_id]
    temp_dataframe['month'] = [month]
    temp_dataframe['avg_spd(km/hour)'] = [raw_avg_distance]
    temp_dataframe['avg_bearing(degree)'] = [raw_avg_bearing]
    temp_dataframe['raw_avg_land_area(sqkm)'] = [raw_avg_land_area]
    temp_dataframe[sbasin_name + "_storm"] = sbasin_binary # 1 means the storm center is in the sub-basin

    temp_dataframe['selected_mtpr(mm/hour)'] = kept_avg_mtpr
    # compute the average moisture variables in the selected storm area
    for varname, xarray in era_field.items():
        # select the time stamp
        subset_xarray = xarray.sel(time=timestamp_range)
        # compute the mean
        subset_mean_xarray = subset_xarray.mean(dim="time")
        # select the var field
        subset_mean_array = subset_mean_xarray[varname].data
        kept_var_array = np.where(kept_mtpr_binary != 0, subset_mean_array, 0.0)

        # compute the area average
        area_avg_var = (kept_var_array * pixel_area).sum() / kept_area
        temp_dataframe["land_" + varname] = [area_avg_var]

        # update the xarray data
        mean_select_daily_storm_nc["land_" + varname].data = kept_var_array

    # compute the residual term
    area_avg_residual = temp_dataframe["land_mtpr"].values[0] + temp_dataframe["land_mer"].values[0] + \
                                      temp_dataframe["land_mvimd"].values[0] + temp_dataframe["land_dwdt"].values[0]
    temp_dataframe['land_residual'] = [area_avg_residual]
    # update record
    residual_array = mean_select_daily_storm_nc["land_mtpr"].data + mean_select_daily_storm_nc["land_mer"].data + \
                     mean_select_daily_storm_nc["land_mvimd"].data + mean_select_daily_storm_nc["land_dwdt"].data
    mean_select_daily_storm_nc['land_residual'].data = residual_array
    mean_select_daily_storm_nc['mask'].data = kept_mtpr_binary

    # save xarray
    mean_select_daily_storm_nc.to_netcdf(
        os.path.join(xarray_save_Loc, storm_id + ".nc"),
            encoding = {"mask": {"dtype": "int16", 'zlib': True},
                    "land_mtpr": {"dtype": "f4", 'zlib': True},
                    "land_mer": {"dtype": "f4", 'zlib': True},
                    "land_mvimd": {"dtype": "f4", 'zlib': True},
                    "land_dwdt": {"dtype": "f4", 'zlib': True},
                    "land_residual": {"dtype": "f4", 'zlib': True}
                    })
    return temp_dataframe


def build_subset_sbasin_storm_catalog(year, area_of_interest, duration_of_interest, sbasin_name):
    """
    Create a subset storm catalog at the basin, where the storms have a specific area and duration.
    :param year: The year of the storm catalog.
    :param area_of_interest: Target area of the region for the storm event
    :param duration_of_interest: Target duration for the storm event
    :param sbasin_name: The sub-basin name.
    :return:
    """
    print("Create subset for sbasin {0} with area {1} sqkm and duration {2} hour in year{3}.".format(sbasin_name, area_of_interest, duration_of_interest, year))
    ts = time()
    # create save folder
    save_sbasin_folder = r"subbasin_catalog" + "\\" + str(sbasin_name)
    make_folder(save_sbasin_folder)
    save_root_folder = os.path.join(save_sbasin_folder, str(year))
    make_folder(save_root_folder)
    # create subset folder
    subset_save_folder = save_root_folder + "/area_{0}_duration_{1}".format(area_of_interest, duration_of_interest)
    make_folder(subset_save_folder)
    # create folder to save xarray of the selected storms
    subset_xarray_save_folder = subset_save_folder + "/xarray"
    make_folder(subset_xarray_save_folder)

    # load precipitation data
    mtpr_xarray = xr.open_dataset("era5_data" + "\\" + "ERA5_hourly_mtpr_processed_2019_2019_1_2.nc")
    # load evaporation data
    mer_xarray = xr.open_dataset("era5_data" + "\\" + "ERA5_hourly_mer_processed_2019_2019_1_2.nc")
    # load divergence data
    mvimd_xarray = xr.open_dataset("era5_data" + "\\" + "ERA5_hourly_mvimd_processed_2019_2019_1_2.nc")
    # load time derivative data
    dwdt_xarray = xr.open_dataset("era5_data" + "\\" + "ERA5_hourly_dwdt_processed_2019_2019_1_2.nc")

    # rename dwdt array from tcwv to dwdt
    # dwdt_xarray = dwdt_xarray.rename({'tcwv':"dwdt"})
    # create a dictionary for the variables
    add_xarray_dict = {
        "mtpr":mtpr_xarray,
        "mer":mer_xarray,
        "mvimd":mvimd_xarray,
        "dwdt":dwdt_xarray,
    }

    root_loc = r"hourly_catalog"
    # load sbasin array
    sbasin_array = np.load(r"boundary_files/{0}_basin_array_boundary.npy".format(sbasin_name))
    # load sbasin catalog #
    sbasin_storm_catalog = pd.read_pickle(os.path.join(root_loc, sbasin_name + "_land_storm_catalog_" + str(year) + ".pkl"))
    # get storm records that satisfy duration requirement
    subset_sbasin_storm_catalog = sbasin_storm_catalog[sbasin_storm_catalog["sbasin_duration(hour)"] >= duration_of_interest]
    # get the unique id
    storm_ids = subset_sbasin_storm_catalog['ID'].unique()
    # storm_num = storm_ids.shape[0]
    # print("Num of storm > {0} hours: {1}".format(duration_of_interest, storm_num))
    # create a subset storm catalog
    subset_storm_catalog = pd.DataFrame()
    for storm_id in storm_ids:
        # print("Processing storm {0}".format(storm_id))
        # extract the single storm stats
        single_storm_stats = subset_sbasin_storm_catalog[subset_sbasin_storm_catalog['ID'] == storm_id]
        # reset the selected index
        single_storm_stats = single_storm_stats.reset_index(drop=True)
        single_storm_xarray = xr.open_dataset(root_loc + "/single_record/" + storm_id + ".nc")
        # run the area duration searcher, get the selected event record and save xarray
        temp_dataframe = area_duration_selector(area_of_interest, duration_of_interest,
                        storm_stats = single_storm_stats, storm_xarray = single_storm_xarray, sbasin_name = sbasin_name,
                        sbasin_array = sbasin_array, xarray_save_Loc = subset_xarray_save_folder, era_field = add_xarray_dict)
        # append to full dataframe
        subset_storm_catalog = subset_storm_catalog.append(temp_dataframe, ignore_index=True)
    # save the subset_storm_catalog
    subset_storm_catalog.to_pickle(os.path.join(subset_save_folder, sbasin_name + "_subset_storm_catalog_{0}.pkl".format(year)))

    print("Finish subset for sbasin {0} with area {1} sqkm and duration {2} hour in year{3}.".format(sbasin_name, area_of_interest, duration_of_interest, year))
    print("Time spent {0} s".format(time()- ts))


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    initial_time = time()
    # year_list = np.arange(1951, 2021)
    # area_list = [5000, 10000, 25000, 50000, 100000]
    # duration_list = [1, 3, 6, 24, 72, 120]
    # sbasin_list = ['miss', 'arkansas', 'low_miss', 'missouri', 'up_miss', 'ohio']

    year_list = [2019]
    area_list = [5000]
    duration_list = [6]
    sbasin_list = ['miss']

    for year in year_list:
        for sbasin_name in sbasin_list:
            for area_of_interest in area_list:
                for duration_of_interest in duration_list:
                    build_subset_sbasin_storm_catalog(year, area_of_interest, duration_of_interest, sbasin_name)


    print("All task is finished, time spent {0} s".format(time()-initial_time))
