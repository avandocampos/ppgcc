import cv2
import numpy as np
import scipy.stats as st


def low_pass_filter_gaussian(image, kernel_size):
    height, width = image.shape
    filtered_image = image.copy().astype(float)
    offset = kernel_size // 2

    # Create a Gaussian kernel
    x = np.arange(-offset, offset + 1)
    gaussian_kernel = np.outer(st.norm.pdf(x), st.norm.pdf(x))

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            sub_image = image[y - offset:y + offset + 1, x - offset:x + offset + 1]
            filtered_image[y, x] = np.sum(sub_image * gaussian_kernel)

    return filtered_image.astype('uint8')


if __name__ == '__main__':

    image = cv2.imread('imagem.jpeg', 0)  # Load image in grayscale
    filtered_image_gaussian = low_pass_filter_gaussian(image, 5)  # Apply filter with 5x5 kernel
    cv2.imwrite('filtered_image_gaussian.jpeg', filtered_image_gaussian)
