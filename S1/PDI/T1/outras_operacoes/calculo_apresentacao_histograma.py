import cv2
import matplotlib.pyplot as plt


def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
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

    image = cv2.imread('images/sample.jpg', 0)  # Load image in grayscale
    histogram = calculate_histogram(image)  # Calculate histogram
    display_histogram(histogram)  # Display histogram
