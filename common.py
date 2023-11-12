import numpy as np


def get_histogram(image):
    # calculate histogram
    histogram = np.zeros(256)
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1
    return histogram


def calculate_CDF(histogram):
    # Cumulative Disturbation function
    cdf = np.zeros(256)
    cdf[0] = histogram[0]

    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]
    return cdf