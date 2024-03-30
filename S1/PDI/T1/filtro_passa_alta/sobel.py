import cv2
import numpy as np


def high_pass_sobel_filter(image):
    height, width = image.shape
    filtered_image = image.copy().astype(float)
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            sub_image = image[y - 1:y + 2, x - 1:x + 2]
            gx = np.sum(sub_image * sobel_kernel_x)
            gy = np.sum(sub_image * sobel_kernel_y)
            filtered_image[y, x] = np.sqrt(gx**2 + gy**2)

    # Normalize the image to the range 0-255
    filtered_image = filtered_image - filtered_image.min()
    filtered_image = (filtered_image / filtered_image.max()) * 255

    return filtered_image.astype('uint8')


if __name__ == '__main__':

    image = cv2.imread('images/sample.jpg', 0)  # Load image in grayscale
    filtered_image_sobel = high_pass_sobel_filter(image)  # Apply filter
    cv2.imwrite('images/filtered_image_sobel.jpg', filtered_image_sobel)
