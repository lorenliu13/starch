# The storm tracker that tracks storm along the time steps.
# @author:xiaoye
# 2021/12/21


import copy
import os
from math import sqrt
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial.distance import pdist, squareform
from skimage.segmentation import relabel_sequential
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from datetime import datetime
from tqdm import tqdm


def track(grown_array: np.ndarray, prcp_array: np.ndarray, ratio_threshold: float, dry_spell_time: int):
    """
    Storm tracking method that labels consecutive storms over time with the same integer labels. The code is modified
    based on github project Storm Tracking and Evaluation Protocol (https://github.com/RDCEP/STEP,
    author: Alex Rittler.
    :param grown_array: Result array from storm identification with dimension of (time, lon, lat).
    :param prcp_array: Raw precipitation field with dimension of (time, lon, lat).
    :param ratio_threshold: Threshold of overlapping ratio, default is 0.3.
    :param dry_spell_time: Allow method to match storm at the time step of (t-1-dry_spell_time), if no match is found at
    t-1 step, default is 0.
    :return:
    """

    # get total time slice
    num_time_slices = grown_array.shape[0]
    # make a copy of the result of the identification algorithm to avoid labeling collisions
    # we will record any labeling changes here
    result_data = copy.deepcopy(grown_array)

    # skip labeling t=0, since it is already labeled correctly
    # for every other time slice
    for time_index in range(1, num_time_slices):

        # find the labels for this time index and the labeled storms in the previous time index
        current_labels = np.unique(grown_array[time_index])

        # and prepare the corresponding precipitation data
        curr_precip_data = prcp_array[time_index]

        # determine the maximum label already used to avoid collisions
        if time_index == 1:
            max_label_so_far = np.max(result_data[time_index])
        else:
            max_label_so_far = np.max(result_data[:time_index])
        # print time index
        # print("Time slice : {0}".format(time_index))
        # then, for each label in current time index (that isn't the background)
        for label in current_labels:
            if label:
                # print current storm number
                # print(f'Current storm label {0}'.format(label))
                # make sure initially the max storm size and best matched storm are 0
                max_size = 0
                best_matched_storm = 0
                # find where the labels of the current storm segment exist in the current time slice
                current_label = np.where(grown_array[time_index] == label, 1, 0)
                curr_size = np.sum(current_label)

                # find the precipitation data at those locations
                curr_label_precip = np.where(grown_array[time_index] == label, curr_precip_data, 0)

                # and its intensity weighted centroid
                curr_centroid = center_of_mass(curr_label_precip)

                # match storms at forward time steps
                if time_index >= dry_spell_time + 1:
                    # back_step = 1, 2 if dry_spell_time = 1
                    for back_step in np.arange(1, dry_spell_time + 2):
                        # print("Match previous storm at {0}".format(time_index - back_step))
                        max_size, best_matched_storm = storm_match(result_data, prcp_array, max_size,
                                                                   best_matched_storm, time_index, back_step,
                                                                   current_label, curr_size, curr_centroid,
                                                                   ratio_threshold)
                        # if find a match, stop current loop
                        if max_size:
                            break
                else:
                    # if time_index < dry_spell_time
                    back_step = 1
                    max_size, best_matched_storm = storm_match(result_data, prcp_array, max_size,
                                                               best_matched_storm, time_index, back_step,
                                                               current_label, curr_size, curr_centroid,
                                                               ratio_threshold)
                # if we found matches
                if max_size:
                    # link the label in the current time slice with the appropriate storm label in the previous
                    result_data[time_index] = np.where(grown_array[time_index] == label, best_matched_storm,
                                                       result_data[time_index])

                # otherwise we've detected a new storm
                else:
                    # give the storm a unique label
                    result_data[time_index] = np.where(grown_array[time_index] == label, max_label_so_far + 1,
                                                       result_data[time_index])

                    max_label_so_far += 1

    result_data = result_data.astype('int')
    seq_result = relabel_sequential(result_data)[0]

    return seq_result


def displacement(current: np.ndarray, previous: np.ndarray) -> np.array:
    """Computes the displacement vector between the centroids of two storms.
    :param current: the intensity-weighted centroid of the storm in the current time slice, given as a tuple.
    :param previous: the intensity-weighted centroid of the storm in the previous time slice, given as a tuple.
    :return: the displacement vector, as an array.
    """
    return np.array([current[0] - previous[0], current[1] - previous[1]])


def magnitude(vector: np.ndarray) -> float:
    """Computes the magnitude of a vector.
    :param vector: the displacement vector, given as an array.
    :return: its magnitude, as a float.
    """
    return sqrt((vector[0] ** 2) + (vector[1] ** 2))


def storm_match(result_data : np.ndarray, prcp_array : np.ndarray, max_size : float,
                best_matched_storm : int, time_index : int, back_step : int, current_label : int,
                curr_size : int, curr_centroid : tuple, ratio_threshold : float):
    """
    The algorithm that searches the best match previous storm for the current storm.
    :param result_data: Storm identification array.
    :param prcp_array: Raw precipitation array.
    :param max_size: Current matched storm size.
    :param best_matched_storm: ID of the current best matched storm.
    :param time_index: Current time step.
    :param back_step: Backward step number for storm match. Previous time step = time_index - back_step.
    :param current_label: Label of the current storm.
    :param curr_size: Size of the current storm in pixels.
    :param curr_centroid: Centroid of the current storm.
    :param ratio_threshold: Threshold of overlapping ratio
    :return:
    max_size: The size of the best matched storm.
    best_matched_storm: The label of the best matched storm.
    """
    max_ratio = 0
    prev_size = 0
    # get previous storm ids and prcp data
    previous_storms = np.unique(result_data[time_index - back_step])
    prev_precip_data = prcp_array[time_index - back_step]

    for storm in previous_storms:
        if storm == 0:  # skip the background
            continue

        # find the storm location in previous time step
        previous_storm = np.where(result_data[time_index - back_step] == storm, 1, 0)
        prev_size = np.sum(previous_storm)

        # selected the overlap area of current storm to prev storm
        overlap_curr_to_prev = np.where(previous_storm == 1, current_label, 0)

        # compute overlapping size
        overlap_size_curr_to_prev = np.sum(overlap_curr_to_prev)
        # compute the overlapping ratio A/current_storm_size
        overlap_ratio_curr_to_prev = overlap_size_curr_to_prev / curr_size

        # selected the overlap area of prev to curr
        overlap_prev_to_curr = np.where(current_label == 1, previous_storm, 0)

        overlap_size_prev_to_curr = np.sum(overlap_prev_to_curr)
        # compute the overlapping ratio: A/previous_storm_size
        overlap_ratio_prev_to_curr = overlap_size_prev_to_curr / prev_size

        # add the two ratio together = A/current_storm_size + A/previous_storm_size
        integrated_ratio = overlap_ratio_curr_to_prev + overlap_ratio_prev_to_curr

        # find the largest overlapping ratio
        if integrated_ratio > max_ratio:
            max_ratio = integrated_ratio
            temp_matched_storm = storm
    # if the max overlapping ratio is larger than threshold
    if max_ratio > ratio_threshold:

        # prev_storm_precip = np.where(result_data[time_index - back_step] == temp_matched_storm, prev_precip_data, 0)
        # prev_centroid = center_of_mass(prev_storm_precip)
        # curr_prev_displacement = displacement(curr_centroid, prev_centroid)  # compute displacement vector
        # curr_prev_magnitude = magnitude(curr_prev_displacement)  # compute centroid distance in pixel
        # if curr_prev_magnitude < max_distance:
        best_matched_storm = temp_matched_storm
        max_size = prev_size

    return max_size, best_matched_storm
