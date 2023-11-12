 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from common import *

def histogram_matching(input, template):
    # Calculate histograms
    input_hist = get_histogram(input)
    template_hist = get_histogram(template)

    # calculate CDFs
    input_cdf = calculate_CDF(input_hist)
    template_cdf = calculate_CDF(template_hist)

    # Check if CDFs functions are cumulative
    if not (all(np.diff(input_cdf) >= 0) and all(np.diff(template_cdf) >= 0)):
        raise ValueError("Cumulative distribution functions should be monotonically increasing.")

    # Apply Hist-Matching
    lut = np.zeros(256)

    for i in range(256):
        j = 0
        while j < 256 and template_cdf[j] < input_cdf[i]:
            j += 1
        lut[i] = j

    matched_image = lut[input].astype(np.uint8)

    return matched_image, input_cdf, template_cdf

if __name__ == '__main__':

    for input_image_path, template_image_path \
            in [["image1.jpg", "image2.jpg"], ["image2.jpg", "image1.jpg"]]:  # add as many [input, template] images as you need
        input_img = cv2.imread(input_image_path, 0)
        template_img = cv2.imread(template_image_path, 0)

        try:
            matched_img, input_cdf, template_cdf = histogram_matching(input_img, template_img)
            matched_cdf = calculate_CDF(get_histogram(matched_img))

            plt.figure(figsize=(15, 8))

            plt.subplot(231), plt.imshow(input_img, cmap='gray'), plt.title('Input Image')
            plt.subplot(232), plt.imshow(template_img, cmap='gray'), plt.title('Template Image')
            
            # Plotting CDFs
            plt.subplot(233), plt.plot(input_cdf, color='red'), plt.title('Input Image CDF')
            plt.subplot(233), plt.plot(template_cdf, color='blue'), plt.title('Template Image CDF'), plt.legend(['input', 'template'])

            # Plotting Output image, alongside old CDFs and the matched one
            plt.subplot(234), plt.imshow(matched_img, cmap='gray'), plt.title('Matched Image')

            plt.subplot(235), plt.plot(input_cdf, color='red'), plt.title('Input Image CDF')
            plt.subplot(235), plt.plot(template_cdf, color='blue'), plt.title('Template Image CDF')
            plt.subplot(235), plt.plot(matched_cdf, color='green'), plt.title('Matched Image CDF'), plt.legend(['input', 'template', 'output'])
            plt.show()

        except ValueError as e:
            print(f"Error: {e}")
