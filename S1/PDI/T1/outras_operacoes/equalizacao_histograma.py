import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def histogram_equalization(image):
    histogram = calculate_histogram(image)
    cumulative_histogram = np.cumsum(histogram)
    normalized_cumulative_histogram = cumulative_histogram * 255 / cumulative_histogram[-1]
    equalized_image = normalized_cumulative_histogram[image]

    return equalized_image.astype('uint8')


def calculate_histogram(image):
    histogram = np.zeros(256)

    for pixel in image.ravel():
        histogram[pixel] += 1

    return histogram


def display_histogram(histogram):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()


def save_histogram(histogram, filename):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':

    image = Image.open('images/sample_2.jpg').convert('L')  # Load image in grayscale
    image = np.array(image)
    equalized_image = histogram_equalization(image)  # Equalize histogram
    equalized_histogram = calculate_histogram(equalized_image)  # Calculate histogram of equalized image

    # Save original histogram and equalized histogram
    save_histogram(calculate_histogram(image), 'images/original_histogram.png')
    save_histogram(equalized_histogram, 'images/equalized_histogram.png')

    # Save equalized image
    Image.fromarray(equalized_image).save('images/equalized_image.jpg')
