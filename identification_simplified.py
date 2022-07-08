# A simplified identification code modified based on identification.py.
# The identification steps only include 4 major steps:
# 1) filter precipitation by a high threshold, connect contiguous regions;
# 2) dilate storm objects to connect nearby regions;
# 3) erode storm objects to remove small regions;
# 4) grow the storm object to a lower boundary threshold.
# This simplified version works well in identifying large-scale storm systems, e.g., tropical/extra-tropical cyclones,
# atmospheric rivers, and mesoscale convective systems.
# 2022/07/07

# import packages
import copy
import numpy as np
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage import morphology
from skimage import draw
from skimage.segmentation import relabel_sequential
import skimage.segmentation
from tqdm import tqdm


def identification(prcp_array: np.ndarray, high_threshold: float = 0.5, low_threshold: float = 0.03,
                   morph_radius: int = 4, expand_distance: int = 8):
    """
    Method to identify and label storms at each time step from precipitation field.
    :param prcp_array: Precipitation array with dimension of (time, lon, lat).
    :param high_threshold: High precipitation filtering threshold, default 0.5 mm/hour.
    :param low_threshold: Low precipitation growing threshold, default 0.03 mm/hour.
    :param morph_radius: Radius of a circular morph structure for storm pattern dilation and erosion,
                         default 4 pixels.
    :param expand_distance: Maximum distance of dilation when growing storm area, default 8 pixels.
    :return: An array that individual storms are assigned with unique integer labels at each time step.
    """

    # define a morph structure
    morph_structure = build_morph_structure(radius=morph_radius)
    # filter with high threhsold
    high_threshold_data = np.where(prcp_array >= high_threshold, prcp_array, 0)
    low_threshold_data = np.where(prcp_array >= low_threshold, prcp_array, 0)
    # find the dimensions of the input
    shape = prcp_array.shape

    # use 8-connectivity for determining connectedness below
    connectivity = generate_binary_structure(2, 2)

    # 1) run the connected-components algorithm on the data and store it in a new array
    label_array = perform_connected_components(high_threshold_data, connectivity)

    # 2) perform a morph dialation
    filtered_label_array = perform_morph_op(morphology.dilation, label_array, morph_structure)
    # perform connected labeling to merge close components
    filtered_label_array = perform_connected_components(filtered_label_array, connectivity)
    # apply it to raw labelled array
    processed_label_array = np.where((label_array != 0), filtered_label_array, 0)

    # 3) perform a morph erosion
    temp_label_array = copy.deepcopy(processed_label_array)
    eroded_label_array = perform_morph_op(morphology.erosion, temp_label_array, morph_structure)
    # keep only those storm ids in eroded_labeled_array
    unique_storm_labels = np.unique(eroded_label_array)
    unique_storm_labels = unique_storm_labels[unique_storm_labels != 0]
    processed_label_array = np.where(np.isin(processed_label_array, unique_storm_labels), processed_label_array, 0)

    # 4) grow to lower boundary
    expanded_label_array = skimage.segmentation.expand_labels(processed_label_array, distance=expand_distance)
    # intersect with low precipitation boundary
    grown_label_array = np.where(low_threshold_data != 0, expanded_label_array, 0)

    # Return the identified results
    return grown_label_array


def build_morph_structure(morph_radius):
    """
    Create an array that represents a circular morph structure with specific radius.
    :param morph_radius: Radius of the morph structure.
    :return: An array describing the morph structure.
    """
    struct = np.zeros((2 * morph_radius, 2 * morph_radius))
    rr, cc = draw.disk(center=(morph_radius - 0.5, morph_radius - 0.5), radius=morph_radius)
    struct[rr, cc] = 1  # data in the circle equals 1
    return struct


def perform_connected_components(to_be_connected: np.ndarray, result: np.ndarray, lifetime: int,
                                 connectivity_type: np.ndarray) -> None:
    """
    This code is from github project Storm Tracking and Evaluation Protocol (https://github.com/RDCEP/STEP,
    author: Alex Rittler.
    Higher order function used to label connected-components on all time slices of a dataset.
    :param to_be_connected: the data to perform the operation on, given as an array of dimensions Time x Rows x Cols.
    :param result: where the result of the operation will be stored, with the same dimensions as to_be_connected.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param connectivity_type: an array representing the type of connectivity to be used by the labeling algorithm. See
    scipy.ndimage.measurements.label for more information.
    :return: (None - the operation is performed on the result in place.)
    """
    for index in range(lifetime):
        cc_output, _ = label(to_be_connected[index], connectivity_type)  # label also returns # of labels found
        result[index] = cc_output


def perform_morph_op(morph_function: object, to_be_morphed: np.ndarray,
                     result: np.ndarray, lifetime: int, structure: np.ndarray) -> None:
    """
    This code is from github project Storm Tracking and Evaluation Protocol (https://github.com/RDCEP/STEP,
    author: Alex Rittler.
    Higher order function used to perform a morphological operation on all time slices of a dataset.
    :param morph_function: the morphological operation to perform, given as an object (function).
    :param to_be_morphed: the data to perform the operation on, given as an array of dimensions Time x Rows x Cols.
    :param result: where the result of the operation will be store, with the same dimensions as to_be_morphed.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param structure: the structural set used to perform the operation, given as an array. See scipy.morphology for more
    information.
    :return: (None - the operation is performed on the result in place.)
    """
    for index in range(lifetime):
        operation = morph_function(to_be_morphed[index], structure)
        result[index] = operation

