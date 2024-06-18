import numpy as np

def gaussian_filter(image, kernel_size=5, sigma=1.0):
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)), 
            (size, size)
        )
        return kernel / np.sum(kernel)

    kernel = gaussian_kernel(kernel_size, sigma)
    padded_image = np.pad(image, kernel_size // 2, mode='constant')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size])

    return filtered_image


def median_filter(image, kernel_size=3):
    padded_image = np.pad(image, kernel_size // 2, mode='constant')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])

    return filtered_image


def low_pass_filter_fft(image, cutoff_frequency=30):
    def fft2d(image):
        return np.fft.fftshift(np.fft.fft2(image))

    def ifft2d(image):
        return np.abs(np.fft.ifft2(np.fft.ifftshift(image)))

    def create_low_pass_filter(shape, cutoff):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        filter = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - center_row)**2 + (j - center_col)**2) < cutoff:
                    filter[i, j] = 1
        return filter

    f_transform = fft2d(image)
    low_pass_filter = create_low_pass_filter(image.shape, cutoff_frequency)
    filtered_transform = f_transform * low_pass_filter
    filtered_image = ifft2d(filtered_transform)

    return filtered_image
