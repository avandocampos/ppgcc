import numpy as np

def moving_average_threshold(image, window_size):
    padded_image = np.pad(image, window_size // 2, mode='constant')
    thresholded_image = np.zeros_like(image)

    for i in range(window_size // 2, image.shape[0] + window_size // 2):
        for j in range(window_size // 2, image.shape[1] + window_size // 2):
            local_region = padded_image[i - window_size // 2:i + window_size // 2 + 1,
                                        j - window_size // 2:j + window_size // 2 + 1]
            local_mean = np.mean(local_region)
            thresholded_image[i - window_size // 2, j - window_size // 2] = 255 if image[i - window_size // 2, j - window_size // 2] > local_mean else 0

    return thresholded_image
