 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import *


def histogram_equalization(image):
    
    histogram = get_histogram(image)
    cdf = calculate_CDF(histogram)
    
    # Checking if the function is cumulative
    if not all(np.diff(cdf) >= 0):
        raise ValueError("CDF function should be monotonically increasing.")

    # Hist-Equalizing
    equalized_image = np.floor(255 * cdf[image] / cdf[-1]).astype(np.uint8)

    return equalized_image, histogram, get_histogram(equalized_image)


if __name__ == '__main__':

    for image_path in ["image1.jpg", "image2.jpg"]:  # add as many images needed
        img = cv2.imread(image_path, 0)

        try:
            equalized_img, original_hist, equalized_hist = histogram_equalization(img)

            # Plotting images and their histograms
            plt.figure(figsize=(10, 5))

            plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original Image')
            plt.subplot(222), plt.bar(np.arange(256), original_hist, color='blue'), plt.title('Original Histogram')

            plt.subplot(223), plt.imshow(equalized_img, cmap='gray'), plt.title('Equalized Image')
            plt.subplot(224), plt.bar(np.arange(256), equalized_hist, color='blue'), plt.title('Equalized Histogram')

            plt.tight_layout()
            plt.show()

        except ValueError as e:
            print(f"Error: {e}")
