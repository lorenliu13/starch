import xarray as xr
import numpy as np
import identification
import tracking
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as patches
from PIL import Image


def draw_rec(labeled_storms, storm_label, lon_data, lat_data):
    """
    Add rectangular boundary for each identified storm in visualization.
    :param labeled_storms: Array of the tracked storms.
    :param storm_label: Current storm label.
    :param lon_data: Longitude coordinate array.
    :param lat_data: Latitude coordinate array.
    :return:
    """

    # extract coordinates
    storm_coord = np.argwhere(labeled_storms == storm_label)
    # longitudinal boundary
    x_max = lon_data[storm_coord[:, 1].max()]
    x_min = lon_data[storm_coord[:, 1].min()]

    # latitudinal boundary
    y_max = lat_data[storm_coord[:, 0].max()]
    y_min = lat_data[storm_coord[:, 0].min()]

    # Rectangle((x,y), width, height, angle) # draw the rectangle
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, facecolor='none',
                             edgecolor='black')
    plt.gca().add_patch(rect)

    # put text at the centroid
    plt.text((x_min + x_max) / 2, (y_min + y_max) / 2, str(storm_label), horizontalalignment='center',
             fontsize=8)


def sequence_storm_label_plot(labeled_storms, timestamp, lon_data, lat_data, save_location,
                              title="Precipitation intensity"):
    """
    Generate images of tracked storms for each time step.
    :param labeled_storms: Storm tracking result array with dimensions of (times, rows, cols).
    :param timestamp: The time step of storm tracking results.
    :param lon_data: Longitude coordinate array.
    :param lat_data: Latitude coordinate array.
    :param save_location： Image save locations.
    :param title: The title of the saved images.
    """
    # storm visualization
    colors = plt.get_cmap("hsv")
    levels = list(np.arange(0, np.max(labeled_storms) + 1))  # the color levels depend on the number of unique storms
    # levels = list(np.arange(0, 200 + 1))
    # create an empty list to collect images, which can be used to generate animations
    images = []
    print("Generating figures: " + title)

    # create color levels for unique storms
    norm = BoundaryNorm(levels, ncolors=len(levels), clip=True)

    plt.figure(num=1, figsize=(5, 2))
    for time_index in tqdm(range(labeled_storms.shape[0])):
        # plot the storm objects
        plt.pcolormesh(lon_data, lat_data,
                       np.ma.masked_where(labeled_storms[time_index] == 0, labeled_storms[time_index]),
                       cmap=colors, norm=norm, shading='auto')
        # add rectangular boundary and number for each storm
        unique_labels = np.unique(labeled_storms[time_index])
        for i in unique_labels:
            if i == 0:  # skip 0 as the background
                continue
            else:
                draw_rec(labeled_storms[time_index], i, lon_data, lat_data)  # 绘制storm

        plt.axis('equal')
        plt.title(title + " at time " + str(time_index) + "\n" + timestamp[time_index])
        plt.savefig(save_location + "\\" + title + " at time " + str(time_index) + ".png", bbox_inches='tight')
        # append the figure to the list
        with Image.open(save_location + "\\" + title + " at time " + str(time_index) + ".png") as im:
            images.append(np.array(im))
        plt.cla()
    plt.close()
    return images


if __name__ == "__main__":
    # load the processed precipitation xarray
    prcp_xarray = xr.open_dataset("era5_data\ERA5_hourly_mtpr_processed_2019_2019_1_2.nc")
    # extract precipitation data
    prcp_array = prcp_xarray['mtpr'].data
    # extract the coordinates
    lat_data = prcp_xarray['latitude'].data
    lon_data = prcp_xarray['longitude'].data
    time_stamp = prcp_xarray['time'].data.astype("datetime64[h]").astype("str")

    # run double-threshold identification
    identify_array = identification.identification(prcp_array, high_threshold=0.5, low_threshold=0.03,
                                                   morph_radius=4)
    # save identified results
    np.save("identify_array.npy", identify_array)

    # identify_array = np.load("identify_array.npy")
    # run tracking using identified results
    track_array = tracking.track(grown_array=identify_array, prcp_array=prcp_array, ratio_threshold=0.3,
                                 max_distance=15, dry_spell_time=0)
    # save tracking results
    np.save("tracking_array.npy", track_array)

    # visualize tracking results
    # create image save location
    parent_loc = "images"
    # make_folder(img_save_location)
    # generated images
    sequence_storm_label_plot(track_array, timestamp=time_stamp, lon_data=lon_data, lat_data=lat_data,
                              save_location=parent_loc, title="Storm_tracking")
