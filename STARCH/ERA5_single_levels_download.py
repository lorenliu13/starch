# download api for EFR5 single level data
# @author: Yuan Liu
# 2021/06/14

import cdsapi
import datetime
import os
import urllib3


# function to make a folder
def make_folder(save_path):
    try:
        os.mkdir(save_path)
    except OSError as error:
        print("Folder exists.")


def eradownload(file_format: str, folder_out: str, area: list,
                start_year: int, end_year: int, start_month: int, end_month: int,
                variable: str, dataset: str, product_type: str):
    """ Download file from CDS，file name as : "ERA5_hourly_variable+start_year+start_month+end_year+end_month.nc"
    :param file_format: file format "grib" or "netcdf"
    :param folder_out: save path
    :param area: download region [lat max, lon min, lat min, lon max] US: [52, -130, 24, -60, ]
    :param start_year: start year
    :param end_year: end year
    :param start_month: start month
    :param end_month: end month
    :param variable: variable name
    :param dataset: dataset name "reanalysis-era5-single-levels"
    :param product_type: dataset's product type: 'reanalysis'
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # ban the warning
    # process time data
    years = [str(start_year + i) for i in range(end_year - start_year + 1)]
    # months
    months = [str(start_month + i).zfill(2) for i in range(end_month - start_month + 1)]
    # days
    start_day = 1
    end_day = 31
    days = [str(start_day + i).zfill(2) for i in range(end_day - start_day + 1)]

    # join the downloaded file
    download_file_name = "ERA5_hourly_" + variable + "_" + str(start_year) + "_" + str(end_year) + "_" + str(
        start_month) + "_" + str(end_month) + ".nc"
    downloaded_file = os.path.join(folder_out, download_file_name)

    print('Process started. Please wait the ending message ... ')
    start = datetime.datetime.now()  # Start Timer

    c = cdsapi.Client()
    c.retrieve(
        dataset,
        {
            'product_type': product_type,
            'year': years,
            'variable': variable,
            'month': months,
            'day': days,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': area,
            'format': file_format,
        },
        downloaded_file)

    print('Process completed in ', datetime.datetime.now() - start)


if __name__ == "__main__":
    # customization
    year_list = [2019, 2020]
    for year in year_list:
        file_format = "netcdf"  # "grib" or "netcdf"

        # area to extract
        area = [52, -130, 24, -60]  # lat max, lon min, lat min, lon max

        # time period to extract
        start_year = year
        end_year = year
        start_month = 1
        end_month = 2

        # set output folder name
        folder_out = r"E:\Atmosphere\Processed_data\ERA5\ERA5_single_level\1H" + "\\" + str(start_year)
        make_folder(folder_out)

        # variables to extract
        variable_list = ['mean_evaporation_rate', 'mean_total_precipitation_rate',
                         'mean_vertically_integrated_moisture_divergence',
                         'total_column_water_vapour']
        for variable in variable_list:
            if variable == "mean_evaporation_rate":
                continue
            # dataset name
            dataset = "reanalysis-era5-single-levels"  # after 1979
            # before 1979 prelim version
            # dataset = 'reanalysis-era5-single-levels-preliminary-back-extension'

            # product type
            product_type = 'reanalysis'
            eradownload(file_format=file_format, folder_out=folder_out, area=area,
                        start_year=start_year, end_year=end_year, start_month=start_month, end_month=end_month,
                        variable=variable, dataset=dataset, product_type=product_type)
