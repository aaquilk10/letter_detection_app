# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy.signal import find_peaks
from scipy import stats

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

def _compute_var_features(window):
    """
    Computes the variance x, y and z acceleration over the given window. 
    """
    return np.var(window, axis=0)

# TODO: define functions to compute more features

def _compute_rfft_features(window):
    """
    Computes the rfft x, y and z acceleration over the given window.
    """
    rfft = np.fft.rfft(window, axis=0)[0].astype(float)
    return rfft

def _compute_histogram_features(window):
    """
    Generates the histogram of x, y and z acceleration over the given window.
    """
    x_hist = np.asarray(np.histogram(window[:,0], bins=100, density=True)[0])
    y_hist = np.asarray(np.histogram(window[:,1], bins=100, density=True)[0])
    z_hist = np.asarray(np.histogram(window[:,2], bins=100, density=True)[0])
    
    return np.asarray([x_hist.mean(), y_hist.mean(), z_hist.mean()])

def _compute_peaks_features(window):
    """
    Computes the peaks of x, y and z acceleration over the given window.
    """
    x_peaks, _ = find_peaks(window[:,0], height=3)
    y_peaks, _ = find_peaks(window[:,1], height=3)
    z_peaks, _ = find_peaks(window[:,2], height=3)
    
    return np.asarray([len(x_peaks), len(y_peaks), len(z_peaks)])

def _compute_peaks_magnitude_features(window):
    """
    Computes the peaks of x, y and z acceleration over the given window.
    """
    magnitude = []
    for i in window:
        magnitude.append(np.linalg.norm(i))
    peaks, _ = find_peaks(magnitude, height=3)
    return [len(peaks)]

def _compute_entropy_features(window):
    """
    Computes the entropy of x, y and z acceleration over the given window.
    """
    x_ent = np.asarray(np.histogram(window[:,0], bins=100, density=True)[0])
    y_ent = np.asarray(np.histogram(window[:,1], bins=100, density=True)[0])
    z_ent = np.asarray(np.histogram(window[:,2], bins=100, density=True)[0])
    
    x_ent = x_ent * np.log(x_ent)
    y_ent = y_ent * np.log(y_ent)
    z_ent = z_ent * np.log(z_ent)
    
    x_ent = np.nansum(x_ent)
    y_ent = np.nansum(y_ent)
    z_ent = np.nansum(z_ent)
    
    return np.asarray([x_ent, y_ent, z_ent])

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    
    x = []
    feature_names = []

    x.append(_compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    # compute_var_features
    x.append(_compute_var_features(window))
    feature_names.append("x_var")
    feature_names.append("y_var")
    feature_names.append("z_var")

    # compute_rfft_features
    x.append(_compute_rfft_features(window))
    feature_names.append("x_rfft")
    feature_names.append("y_rfft")
    feature_names.append("z_rfft")

    # compute_histogram_features
    # NOT USING BECAUSE IT DOESN'T WORK PORPERLY
    # FIXED
    x.append(_compute_histogram_features(window))
    feature_names.append("x_hist")
    feature_names.append("y_hist")
    feature_names.append("z_hist")

    # compute_peaks_features
    # NOT USING BECAUSE IT DOESN'T WORK PORPERLY
    # FIXED
    x.append(_compute_peaks_features(window))
    feature_names.append("x_peaks")
    feature_names.append("y_peaks")
    feature_names.append("z_peaks")
    
    # compute_peaks_magnitude_features
    x.append(_compute_peaks_magnitude_features(window))
    feature_names.append("peaks_magnitude")
    
    # compute_entropy_features
    # NOT USING BECAUSE IT DOESN'T WORK PORPERLY
    # FIXED
    x.append(_compute_entropy_features(window))
    feature_names.append("x_entropy")
    feature_names.append("y_entropy")
    feature_names.append("z_entropy")

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector

print("features.py executed properly")
