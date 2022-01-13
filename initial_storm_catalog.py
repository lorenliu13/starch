# Create hourly land and full storm catalog for the Contiguous United States (CONUS).
# Full storm catalog includes storm precipitation and area over the sea, while the land storm catalog only includes
# storms over the land of CONUS.
# author: Yuan Liu
# 2021/12/21

import numpy as np
from time import time
import pandas as pd
import os
import mpu
from geographiclib.geodesic import Geodesic
import xarray as xr


def make_folder(save_path):
    """
    Function to create a new folder at specified path.
    :param save_path: The path of the new folder.
    :return:
    """
    try:
        os.mkdir(save_path)
    except OSError as error:
        pass


def build_storm_catalog(year, low_threshold, track_storms, mtpr_xarray, mer_xarray,
                        mvimd_xarray, dwdt_xarray, conus_boundary, save_loc):
    """
    Compute first-stage storm attributes based on tracking results.
    :param year: The year of storm tracking.
    :param low_threshold: The low threshold used in storm identification, default 0.03 mm/hour.
    :param track_storms: Storm tracking result array with dim (time, lon, lat).
    :param mtpr_xarray: Preprocessed precipitation nc file read by xarray package.
    :param mer_xarray: Preprocessed evaporation nc file read by xarray package.
    :param mvimd_xarray: Preprocessed divergence nc file read by xarray package.
    :param dwdt_xarray: Preprocessed time derivative of precipitable water nc file read by xarray package
    :param conus_boundary: Boundary array of the CONUS, where the grid is 1 if it is inside the CONUS.
    :param save_loc: Location to save storm catalog data.
    :return:
    """
    # record start time
    ts = time()
    # create a dataframe to store all storms
    full_storm_catalog = pd.DataFrame()
    # create a dataframe to store only the land part of the storms
    land_storm_catalog = pd.DataFrame()

    # extract the data array
    mer_array = mer_xarray['mer'].data
    mvimd_array = mvimd_xarray['mvimd'].data
    # dw/dt array data need to be shortened to have consistent time step
    if year == 1979:
        # note: For ERA5 data in 1979, precipitation, evaporation, and divergence start from T07,
        # but dw/dt starts from T00, so dwdt should be cut to start from T07 to be consistent
        dwdt_array = dwdt_xarray['dwdt'].data[7:-1]
    else:
        # drop the last element to be consistent with other variables.
        dwdt_array = dwdt_xarray['dwdt'].data[:-1]
    # extract the timestamp and covert to string type
    timestamp = mtpr_xarray['time'].data.astype("datetime64[h]").astype("str")
    # extract lon and lat coordinates
    lat_data = mtpr_xarray['latitude'].data
    lon_data = mtpr_xarray['longitude'].data
    # compute projected area of each pixel (km^2)
    lon_2, lat_2 = np.meshgrid(lon_data, lat_data)
    grid_cell_degree = 0.25
    pixel_area = np.cos(lat_2 * np.pi / 180) * 111 * 111 * grid_cell_degree * grid_cell_degree
    # extract precipitation data and filter it by the low threshold
    prcp_array = mtpr_xarray['mtpr'].data
    filtered_array = np.where(prcp_array < low_threshold, 0, prcp_array)
    # quantification
    # quantify storm sizes (sqkm)
    sizes = get_size_prj(storms=track_storms, grid_cell_degree=0.25, lat_data=lat_data, lon_data=lon_data)
    # quantify storm avg intensity (mm/hour)
    averages = get_average(storms=track_storms, precip=filtered_array)
    # quantify storm max intensity (mm/hour)
    max_intensity = get_max_intensity(storms=track_storms, precip=filtered_array)
    # quantify storm central location (degree)
    central_loc = get_central_loc_degree(storms=track_storms, precip=filtered_array,
                                         lat_data=lat_data, lon_data=lon_data)
    # find individual storm ids
    unique_labels = np.unique(track_storms)
    print("Total storm number: {0}".format(unique_labels.shape[0]))
    # create a folder to save spatial pattern of each storm record
    single_record_save_loc = os.path.join(save_loc, "single_record")
    make_folder(single_record_save_loc)

    # skip 0 because 0 is background
    for storm_label in np.arange(1, unique_labels.max() + 1):
        # find time period of the current storm
        storm_binary = sizes[:, storm_label] != 0
        # extract the storm period from tracking results
        storm_tracks = track_storms[storm_binary]
        # assign 1 to the current storm area, 0 otherwise
        selected_storm = np.where(storm_tracks == storm_label, storm_label, 0)

        # extract the precipitation distrbution, mer distribution, mvimd distribution, and dwdt distribution
        precip_distribution = np.where(selected_storm == storm_label, filtered_array[storm_binary], 0)
        mer_distribution = np.where(selected_storm == storm_label, mer_array[storm_binary], 0)
        mvimd_distribution = np.where(selected_storm == storm_label, mvimd_array[storm_binary], 0)
        dwdt_distribution = np.where(selected_storm == storm_label, dwdt_array[storm_binary], 0)

        # compute the storm duration (hour)
        duration_storm = selected_storm.shape[0]
        duration_storm_list = [duration_storm] * duration_storm
        # compute the duration by day
        rounded_days = round(duration_storm / 24)
        rounded_days_list = [rounded_days] * duration_storm
        # extract timestamp sequence
        time_stamp_storm = timestamp[storm_binary]
        time_stamp_storm = pd.DatetimeIndex(time_stamp_storm)
        # initialize lists to save storm centroids
        lon_storm = []
        lat_storm = []
        # initialize the list to save move distance storm centroid per hour (km)
        distance_list = []
        # initialize the list to save storm bearing (degree)
        bearing_list = []
        # extract storm centroids array([lon, lat])
        centroid_coord_pair = central_loc[storm_binary][:, storm_label]
        for i in range(centroid_coord_pair.shape[0]):
            lon_storm.append(centroid_coord_pair[i][0])
            lat_storm.append(centroid_coord_pair[i][1])
            if i == 0:
                # the distance and bearing are 0 for the first time step
                distance = 0
                bearing = 0
                distance_list.append(distance)
                bearing_list.append(bearing)
            else:
                # obtain the centroids for current and previous time steps
                lat1 = centroid_coord_pair[i-1][1]
                lon1 = centroid_coord_pair[i-1][0]
                lat2 = centroid_coord_pair[i][1]
                lon2 = centroid_coord_pair[i][0]
                # compute the distance between two centroids (sqkm)
                # https: // stackoverflow.com / questions / 19412462 / getting - distance - between - two - points - based - on - latitude - longitude
                distance = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
                distance_list.append(distance)
                # compute the bearing between two centroids (degree)
                # https://stackoverflow.com/questions/54873868/python-calculate-bearing-between-two-lat-long
                bearing = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)['azi1']  # return the bearing in degrees
                bearing_list.append(bearing)

        # compute avg intensity (mm/hour)
        avg_intensity_storm = averages[storm_binary, storm_label]
        # compute max intensity (mm/hour)
        max_intensity_storm = max_intensity[storm_binary, storm_label]
        # compute projected area (km^2)
        prj_area_storm = sizes[storm_binary, storm_label]
        # generate storm ID
        id_storm = str(year) + str(storm_label).zfill(5)
        id_storm_list = [id_storm] * duration_storm

        # create lists to save distributions of storm moisture variables
        list_mask = [selected_storm[i] for i in np.arange(duration_storm)]  # the mask corresponds to storm labels
        # precipitation distribution
        list_precip_distribution = [precip_distribution[i] for i in np.arange(duration_storm)]
        # evaporation distribution
        list_mer_distribution = [mer_distribution[i] for i in np.arange(duration_storm)]
        # divergence distribution
        list_mvimd_distribution = [mvimd_distribution[i] for i in np.arange(duration_storm)]
        # time derivative distribution
        list_dwdt_distribution = [dwdt_distribution[i] for i in np.arange(duration_storm)]

        # initialize lists to save area-weighted average moisture variables
        land_avg_mtpr_list = []
        land_avg_mer_list = []
        land_avg_dwdt_list = []
        land_avg_mvimd_list = []
        land_avg_residual_list = []

        # initialize lists to save only the land part of the moisture distribution of the storm
        land_mtpr_distribution_list = []
        land_mer_distribution_list = []
        land_mvimd_distribution_list = []
        land_dwdt_distribution_list = []
        land_residual_distribution_list = []

        # initialize the list to save storm land boundary
        storm_extent_list = []
        # initialize the list to save storm area over the land
        land_area_list = []

        # compute the land extent properties of the storm
        for time_index in np.arange(duration_storm):
            # extract the storm extent
            storm_extent = list_precip_distribution[time_index] != 0
            # extract the storm event over the land
            non_nan_loc = (conus_boundary == 1) & (storm_extent)
            # compute the land area (sqkm)
            non_nan_extent_area = pixel_area[non_nan_loc].sum()
            # append the land extend and arae to the lists
            storm_extent_list.append(non_nan_loc)
            land_area_list.append(non_nan_extent_area)

        # get the moisture variable distributions over the land part of the storm
        for time_index in np.arange(duration_storm):
            land_mtpr_distribution = np.where(storm_extent_list[time_index], list_precip_distribution[time_index], 0)
            land_mer_distribution = np.where(storm_extent_list[time_index], list_mer_distribution[time_index], 0)
            land_mvimd_distribution = np.where(storm_extent_list[time_index], list_mvimd_distribution[time_index], 0)
            land_dwdt_distribution = np.where(storm_extent_list[time_index], list_dwdt_distribution[time_index], 0)
            # compute the residual distribution
            land_residual_distribution = land_mtpr_distribution + land_mer_distribution + land_mvimd_distribution + land_dwdt_distribution
            # append the distribution to the list
            land_mtpr_distribution_list.append(land_mtpr_distribution)
            land_mer_distribution_list.append(land_mer_distribution)
            land_mvimd_distribution_list.append(land_mvimd_distribution)
            land_dwdt_distribution_list.append(land_dwdt_distribution)
            land_residual_distribution_list.append(land_residual_distribution)

            # compute the area-weighted averaged moisture variables
            # if land area is zero, then the average is zero
            if land_area_list[time_index] == 0:
                land_avg_mtpr = 0
                land_avg_mer = 0
                land_avg_mvimd = 0
                land_avg_dwdt = 0
                land_avg_residual = 0
            else:
                # compute the area-weighted averages
                land_avg_mtpr = ((land_mtpr_distribution * pixel_area).sum()) / land_area_list[time_index]
                land_avg_mer = ((land_mer_distribution * pixel_area).sum()) / land_area_list[time_index]
                land_avg_mvimd = ((land_mvimd_distribution * pixel_area).sum()) / land_area_list[time_index]
                land_avg_dwdt = ((land_dwdt_distribution * pixel_area).sum()) / land_area_list[time_index]
                land_avg_residual = land_avg_mtpr + land_avg_mer + land_avg_mvimd + land_avg_dwdt
            # append the averages to the lists
            land_avg_mtpr_list.append(land_avg_mtpr)
            land_avg_mer_list.append(land_avg_mer)
            land_avg_dwdt_list.append(land_avg_dwdt)
            land_avg_mvimd_list.append(land_avg_mvimd)
            land_avg_residual_list.append(land_avg_residual)

        # save storm season and month, which are determined by the first timestamp of the storm
        start_month = time_stamp_storm[0].month
        if start_month in [12, 1, 2]:
            season_storm = "win"
            season_id = 4
        elif start_month in [3, 4, 5]:
            season_storm = 'spr'
            season_id = 1
        elif start_month in [6, 7, 8]:
            season_storm = 'sum'
            season_id = 2
        else:
            season_storm = 'fal'
            season_id = 3
        season_id_list = [season_id] * duration_storm
        season_storm_list = [season_storm] * duration_storm
        month_storm = time_stamp_storm[0].month
        mont_storm_list = [month_storm] * duration_storm

        # create a dataframe to save all the information for the storm
        storm_record = pd.DataFrame()
        storm_record['ID'] = id_storm_list
        storm_record['Projected_area(sqkm)'] = prj_area_storm
        storm_record['Timestamp'] = time_stamp_storm
        storm_record['Avg_intensity(mm/h)'] = avg_intensity_storm
        storm_record['Max_intensity(mm/h)'] = max_intensity_storm
        storm_record['Duration(hour)'] = duration_storm_list
        storm_record['DurationDays(day)'] = rounded_days_list
        storm_record['Central_lon(degree)'] = lon_storm
        storm_record['Central_lat(degree)'] = lat_storm

        storm_record['Season'] = season_storm_list
        storm_record['Season_id'] = season_id_list
        storm_record['Month'] = mont_storm_list
        storm_record['Distance(km)'] = distance_list
        storm_record['Bearing(degree)'] = bearing_list

        storm_record['Land_area(sqkm)'] = land_area_list
        storm_record['Land_area_avg_mtpr(mm/h)'] = land_avg_mtpr_list
        storm_record['Land_area_avg_mer(mm/h)'] = land_avg_mer_list
        storm_record['Land_area_avg_dwdt(mm/h)'] = land_avg_dwdt_list
        storm_record['Land_area_avg_mvimd(mm/h)'] = land_avg_mvimd_list
        storm_record['Land_area_avg_residual(mm/h)'] = land_avg_residual_list

        # deep copy the storm record
        land_storm_record = storm_record.copy(deep=True)
        # adjust the storm duration based on its presence on land
        record_length = land_storm_record.shape[0]
        start_index = 0
        for i in range(record_length):
            if land_storm_record.loc[i, 'Land_area(sqkm)'] == 0:
                land_storm_record.drop(i, axis=0, inplace=True)
                start_index = start_index + 1
            else:
                break
        # if the storm has no time on land, remove the record
        if land_storm_record.shape[0] == 0:
            # print('The storm is empty.')
            continue
        end_index = record_length
        for i in range(record_length):
            if land_storm_record.loc[record_length - 1 - i, 'Land_area(sqkm)'] == 0:
                # print('Remove sea storm record')
                land_storm_record.drop(record_length - 1 - i, axis=0, inplace=True)
                end_index = end_index - 1
            else:
                break
        # if the storm has no time on land, remove the record
        if land_storm_record.shape[0] == 0:
            # print('The storm is empty.')
            continue

        # update the storm duration
        storm_duration = land_storm_record.shape[0]
        storm_day = round(storm_duration / 24)
        storm_duration_list = [storm_duration] * storm_duration
        storm_day_list = [storm_day] * storm_duration
        land_storm_record['Duration(hour)'] = storm_duration_list
        land_storm_record['DurationDays(day)'] = storm_day_list

        # transform the list of moisture distributions to array
        mask_grid_array = np.array(list_mask[start_index:end_index])
        land_mtpr_array = np.array(land_mtpr_distribution_list[start_index:end_index])
        land_mer_array = np.array(land_mer_distribution_list[start_index:end_index])
        land_mvimd_array = np.array(land_mvimd_distribution_list[start_index:end_index])
        land_dwdt_array = np.array(land_dwdt_distribution_list[start_index:end_index])
        land_residual_array = np.array(land_residual_distribution_list[start_index:end_index])

        # save the array to a netcdf file
        da = xr.Dataset(
            data_vars={
                "mask": (('time', "lat", "lon"), mask_grid_array),
                "land_mtpr": (('time', "lat", "lon"), land_mtpr_array),
                "land_mer": (('time', "lat", "lon"), land_mer_array),
                "land_mvimd": (('time', "lat", "lon"), land_mvimd_array),
                "land_dwdt": (('time', "lat", "lon"), land_dwdt_array),
                "land_residual": (('time', "lat", "lon"), land_residual_array),
            },
            coords={
                "lon": lon_data,
                "lat": lat_data,
                "time": time_stamp_storm[start_index:end_index]
            },
            attrs=dict(
                description="Single record of storm " + id_storm,
                units="mm/hour")
        )
        da.to_netcdf(os.path.join(single_record_save_loc, id_storm + ".nc"),
                     encoding={"mask": {"dtype": "int16", 'zlib': True},
                               "land_mtpr": {"dtype": "f4", 'zlib': True},
                               "land_mer": {"dtype": "f4", 'zlib': True},
                               "land_mvimd": {"dtype": "f4", 'zlib': True},
                               "land_dwdt": {"dtype": "f4", 'zlib': True},
                               "land_residual": {"dtype": "f4", 'zlib': True},
                               })

        # append the record to the catalog
        land_storm_catalog = land_storm_catalog.append(land_storm_record, ignore_index=True)
        full_storm_catalog = full_storm_catalog.append(storm_record, ignore_index=True)

    # save the storm catalog
    land_storm_catalog.to_pickle(os.path.join(save_loc, "land_storm_catalog_" + str(year) + ".pkl"))
    full_storm_catalog.to_pickle(os.path.join(save_loc, "full_storm_catalog_" + str(year) + ".pkl"))
    print("Storm catalog in {0} finished, time spent: {1} s".format(year, time() - ts))


def get_duration(storms: np.ndarray, time_interval: float) -> np.ndarray:
    """Computes the duration (in the time unit of time_interval) of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param time_interval: the period between temporal 'snapshots', given as a float.
    :return: An array of length equal to the number of tracked storms + 1, where the value at [x] corresponds to
    the duration of the storm x. The index 0 (referring to the background) is always 0 and provided for ease of
    indexing.
    """
    # find the number of time slices in the data
    lifetime = storms.shape[0]

    ls = []

    for time_index in range(lifetime):
        # compute the labels that appear in that time slice

        curr_labels = np.unique(storms[time_index])

        ls.append(curr_labels)

    # Convert list of different size into a numpy array

    storm_array = np.zeros([len(ls), len(max(ls, key=lambda x: len(x)))])

    for i, j in enumerate(ls):
        storm_array[i][0:len(j)] = j

    storm_array = np.array(storm_array, dtype=np.int32)

    unique, counts = np.unique(storm_array, return_counts=True)

    counts[0] = 0

    result = counts * time_interval

    return result


def get_size_prj(storms: np.ndarray, grid_cell_degree: float, lat_data: np.ndarray, lon_data: np.ndarray) -> np.ndarray:
    """
    Compute the size of each storm with unit of  km^2.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param grid_cell_degree: 0.25 degree for ERA5 storms.
    :param lat_data: latitude coordinate array.
    :param lon_data: longitude coordinate array.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the size of the storm at t=y,
    storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # compute the projected area for each pixel
    lon_2, lat_2 = np.meshgrid(lon_data, lat_data)
    pixel_area = np.cos(lat_2 * np.pi / 180) * 111 * 111 * grid_cell_degree * grid_cell_degree

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:
            if label:
                # add up its coverage area over the pixel_area_matrix
                storm_size = np.sum(np.where(storms[time_index] == label, pixel_area, 0))

                # and place it at that correct location in the array to return
                result[time_index][label] = storm_size

    return result


def get_average(storms: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """
    Computes the average intensity of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip: the precipitation data corresponding to the tracked storms, with the same dimensions as
    tracked_storms.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the mean intensity of the
    storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:

            if label:
                # find the precipitation where it appears in the current time slice
                storm_precip = np.where(storms[time_index] == label, precip[time_index], 0)

                # sum the precipitation
                storm_precip_sum = np.sum(storm_precip)

                # find the number of grid cells belonging to the storm
                storm_size = np.sum(np.where(storms[time_index] == label, 1, 0))

                # find the storm's average precipitation in this time slice
                storm_avg = storm_precip_sum / storm_size

                # and store it in the appropriate place in our result array
                result[time_index][label] = storm_avg

    return result


def get_max_intensity(storms: np.ndarray, precip: np.ndarray) -> np.ndarray:
    """
    Computes the average intensity of each storm across all time slices given.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip: the precipitation data corresponding to the tracked storms, with the same dimensions as
    tracked_storms.
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the mean intensity of the
    storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # find the number of time slices in the data
    lifetime = storms.shape[0]

    # and the number of storms
    total_storms = len(np.unique(storms))

    # initialize an array with dimensions number of time slices by number of storms
    result = np.zeros((lifetime, total_storms))

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        # for each label that appears in this time slice (that's not the background)
        for label in labels:

            if label:
                # find the precipitation where it appears in the current time slice
                storm_precip = np.where(storms[time_index] == label, precip[time_index], 0)

                # get the maximum precipitation
                storm_precip_max = np.max(storm_precip)

                # find the number of grid cells belonging to the storm
                # storm_size = np.sum(np.where(storms[time_index] == label, 1, 0))

                # find the storm's average precipitation in this time slice
                # storm_avg = storm_precip_sum / storm_size

                # and store it in the appropriate place in our result array
                result[time_index][label] = storm_precip_max

    return result


def get_central_loc_degree(storms: np.ndarray, precip: np.ndarray, lat_data: np.ndarray, lon_data: np.ndarray) \
        -> np.ndarray:
    """
    Compute the precipitation intensity weighted centroid of the storm with unit of degree.
    :param storms: the tracked storms returned by the tracking algorithm, given as an array of dimensions
    Time x Rows x Cols.
    :param precip: the precipitation data corresponding to the tracked storms data, with the same dimensions as
    tracked_storms.
    :param lat_data: lat_data. 1 * lenth array
    :param lon_data: lon_data 1 * lenth array
    :return: a lifetime x total_storms array where the value found at [y][x] corresponds to the central location of the
    storm at t=y, storm=x. Except in the case of index 0, which is always 0 for any t.
    """

    # create mesh grid of lat and lon data
    lon_array, lat_array = np.meshgrid(lon_data, lat_data)

    lifetime = storms.shape[0]

    total_storms = len(np.unique(storms))

    # initialize an array to store our result, but of type object to allow us to store an array in each cell
    result = np.zeros((lifetime, total_storms)).astype(object)

    # create arrays of x, y, and z values for the cartesian grid in R3

    # create an array to hold each central location as we calculate it
    central_location = np.empty(2)

    for time_index in range(lifetime):
        # find the unique labels
        labels = np.unique(storms[time_index])

        for label in labels:
            # if the storm exists in this time slice
            if label:
                # find the sum of the precipitation values belonging to the storm
                sum_precipitation = np.sum(np.where(storms[time_index] == label, precip[time_index], 0))

                # and its intensity weighted centroid
                x_avg = np.sum(np.where(storms[time_index] == label, ((lon_array * precip[time_index]) /
                                                                      sum_precipitation), 0))

                y_avg = np.sum(np.where(storms[time_index] == label, ((lat_array * precip[time_index]) /
                                                                      sum_precipitation), 0))

                # get the corresponding lat and lon data
                central_location[0] = x_avg
                central_location[1] = y_avg

                # and we place it in the appropriate spot in the array
                result[time_index][label] = central_location

                # reset the central location - this seems to be necessary here
                central_location = np.zeros(2)
    return result


if __name__ == "__main__":
    # define save location
    save_loc = "hourly_catalog"
    make_folder(save_loc)
    # load tracking data
    tracked_storm_loc = "storm_tracking_results/tracking_array.npy"
    track_storms = np.load(tracked_storm_loc, allow_pickle=True)
    # load precipitation data
    raw_data_folder = "era5_data"
    mtpr_xarray = xr.open_dataset(raw_data_folder + "\\" + "ERA5_hourly_mtpr_processed_2019_2019_1_2.nc")
    # load evaporation data
    mer_xarray = xr.open_dataset(raw_data_folder + "\\" + "ERA5_hourly_mer_processed_2019_2019_1_2.nc")
    # load divergence data
    mvimd_xarray = xr.open_dataset(raw_data_folder + "\\" + "ERA5_hourly_mvimd_processed_2019_2019_1_2.nc")
    # load time derivative data
    dwdt_xarray = xr.open_dataset(raw_data_folder + "\\" + "ERA5_hourly_dwdt_processed_2019_2019_1_2.nc")

    # set parameters
    year = 2019
    low_threshold = 0.03
    conus_boundary = np.load("boundary_files/conus_boundary.npy")
    build_storm_catalog(year, low_threshold, track_storms, mtpr_xarray, mer_xarray,
                        mvimd_xarray, dwdt_xarray, conus_boundary, save_loc)