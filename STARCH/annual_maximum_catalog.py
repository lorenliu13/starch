# Create annual maximum storm catalog
# @author: Yuan Liu
# 2022/01/12

import numpy as np
import pandas as pd
from initial_storm_catalog import make_folder


def annual_maximum_selector(area_of_interest, duration_of_interest, sbasin_name):
    # select annual_maxima of subset storm catalog from 1951 to 2020
    print("Find annual maximum for sbasin {0} with area {1} sqkm and duration {2} hour.".format(sbasin_name,
                                                                                                     area_of_interest,
                                                                                                     duration_of_interest))
    save_sbasin_folder = "annual_maximum_catalog/" + str(sbasin_name)
    make_folder(save_sbasin_folder)
    # create subset folder
    subset_save_folder = save_sbasin_folder + "/area_{0}_duration_{1}".format(area_of_interest, duration_of_interest)
    make_folder(subset_save_folder)

    full_subset_catalog = pd.DataFrame()
    annual_max_subset_catalog = pd.DataFrame()

    year_list = np.arange(1951, 2021)
    for i in range(year_list.shape[0]):
        # for each year
        year = year_list[i]
        # load the corresponding storm catalog
        root_folder = "subbasin_catalog" + "/" + sbasin_name + "/" + str(year) + "/" + "area_{0}_duration_{1}".format(
            area_of_interest, duration_of_interest)
        subset_catalog_stats = pd.read_pickle(root_folder + "/" + sbasin_name + "_subset_storm_catalog_" + str(year) + ".pkl")
        # only keep storms inside the sub-basin
        subset_catalog_stats = subset_catalog_stats[subset_catalog_stats[sbasin_name + "_storm"] == 1]
        # add a column of the storm year
        subset_catalog_stats['year'] = [year] * subset_catalog_stats.shape[0]
        # find the record with the largest precipitation each year
        try:
            annual_maximum_record = subset_catalog_stats[subset_catalog_stats['land_mtpr'] == subset_catalog_stats['land_mtpr'].max()]
        except:
            print("Year {0} has no record.".format(year))
            continue
        # append the full records
        full_subset_catalog = full_subset_catalog.append(subset_catalog_stats, ignore_index=True)
        # append annual maximum record
        annual_max_subset_catalog = annual_max_subset_catalog.append(annual_maximum_record, ignore_index=True)

    # save the annual maximum record to csv and pickle
    annual_max_subset_catalog.to_csv(subset_save_folder + "/" + "annual_maximum_subset_storm_catalog.csv", index=False)
    annual_max_subset_catalog.to_pickle(subset_save_folder + "/" + "annual_maximum_subset_storm_catalog.pkl")
    # save full record to pickle
    full_subset_catalog.to_pickle(subset_save_folder + "/" + "full_subset_storm_catalog.pkl")


if __name__ == "__main__":
    # set up the parameters
    year_list = np.arange(1951, 2021)
    area_list = [25000]
    duration_list = [2]
    sbasin_list = ['arkansas']

    # get annual maximum of each case
    for sbasin_name  in sbasin_list:
        for area_of_interest in area_list:
            for duration_of_interest in duration_list:
                annual_maximum_selector(area_of_interest, duration_of_interest, sbasin_name)

