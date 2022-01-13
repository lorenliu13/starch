# Preprocessing the raw ERA5 data. The unit of precipitation, evaporation, and divergence will change from mm/s to mm/h.
# Compute the time derivative term dw/dt based on total column water vapor using central differencing.
# @ author: Yuan Liu
# 2022/01/10

import xarray as xr
import numpy as np


def time_difference(file_location, save_location):
    """
    Compute the time derivative of total water vapor column using central differencing.
    :param file_location: The location of the raw ERA5 total column water vapor data.
    :param save_location: The location to save the processed data.
    :return:
    """
    print("Preprocessing file \n {0}".format(file_location))
    # read the raw ERA5 data
    xarray = xr.open_dataset(file_location)
    xarray = xarray.isel(time=np.arange(0, 100)) # extract only 100 time step to create a small data set
    # extract the short name of the variable in the data
    short_name = [k for k in xarray.data_vars.keys()]
    short_name = short_name[0]
    # compute central difference (dw/dt) unit: (mm/hour)
    # https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    xarray_grad = np.gradient(xarray[short_name].data, axis=0)
    # save the new time derivative data to xarray
    xarray[short_name].data = xarray_grad
    xarray[short_name].attrs['units'] = 'mm/hour'
    xarray[short_name].attrs['long_name'] = 'dw/dt'
    # rename xarray short name from tcwv to dwdt
    xarray = xarray.rename({'tcwv': "dwdt"})
    # save the xarray
    xarray.to_netcdf(save_location, encoding = {"dwdt":{"dtype" : "float32"}})


def mean_rate_data_preprocessing(file_location, save_location):
    """
    Change the unit of the variable in ERA5 from mm/s to mm/hour.
    :param file_location: The location of the raw ERA5 total column water vapor data.
    :param save_location: The location to save the processed data.
    :return:
    """
    print("Preprocessing file \n {0}".format(file_location))
    # read the raw ERA5 data
    xarray = xr.open_dataset(file_location)
    xarray = xarray.isel(time=np.arange(0, 100)) # extract only 100 time step to create a small data set
    # extract the short name of the variable in the data
    short_name = [k for k in xarray.data_vars.keys()]
    short_name = short_name[0]
    # shift forward one hour
    xarray_sft = xarray.shift(time=-1).dropna(dim='time',how='all')
    # change unit from mm/s to mm/h
    xarray_sft[short_name].data = xarray_sft[short_name].data * 3600
    # rename unit
    xarray_sft[short_name].attrs['units'] = 'mm/h'
    # save the processed file
    xarray_sft.to_netcdf(save_location, encoding={short_name: {"dtype": "float32"}})
    return xarray_sft


if __name__ == "__main__":
    # add the file name
    # note that the file folder and file name should be customized based on downloaded ERA5 data, which are not provided
    # in the code examples
    raw_data_folder = r"E:\Atmosphere\Processed_data\ERA5\ERA5_single_level\1H\2019"
    # run the central difference on total column water vapor
    file_location = raw_data_folder + "\\" + "ERA5_hourly_total_column_water_vapour_2019_2019_1_2.nc"
    save_location = "era5_data\ERA5_hourly_dwdt_processed_2019_2019_1_2.nc"
    time_difference(file_location, save_location)

    # change the unit of evaporation, divergence, and precipitation
    variable_list = ['mean_evaporation_rate', "mean_total_precipitation_rate", "mean_vertically_integrated_moisture_divergence"]
    short_name_list = ['mer', 'mtpr', 'mvimd']
    for i in range(len(variable_list)):
        file_location = raw_data_folder + "\\" + "ERA5_hourly_" + variable_list[i] + "_2019_2019_1_2.nc"
        save_location = "era5_data\ERA5_hourly_" + short_name_list[i] + "_processed_2019_2019_1_2.nc"
        mean_rate_data_preprocessing(file_location, save_location)

