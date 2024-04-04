import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


if __name__ == '__main__':

    image = Image.open('images/sample.jpg').convert('L')  # Load image in grayscale
    image = np.array(image)
    histogram = calculate_histogram(image)  # Calculate histogram
    display_histogram(histogram)  # Display histogram
