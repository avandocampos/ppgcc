import cv2
import numpy as np


def low_pass_filter_median(image, kernel_size):
    height, width = image.shape
    filtered_image = image.copy()
    offset = kernel_size // 2

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            sub_image = image[y - offset:y + offset + 1, x - offset:x + offset + 1]
            filtered_image[y, x] = np.median(sub_image)

    return filtered_image


if __name__ == '__main__':

    image = cv2.imread('images/sample.jpg', 0)  # Load image in grayscale
    filtered_image_median = low_pass_filter_median(image, 5)  # Apply filter with 5x5 kernel
    cv2.imwrite('images/filtered_image_median.jpg', filtered_image_median)
