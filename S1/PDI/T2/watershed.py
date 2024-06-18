import numpy as np
from scipy.ndimage import distance_transform_edt

def watershed_segmentation(image):
    def find_local_maxima(distance):
        local_maxima = np.zeros_like(distance, dtype=bool)
        for i in range(1, distance.shape[0] - 1):
            for j in range(1, distance.shape[1] - 1):
                if distance[i, j] > max(distance[i-1, j], distance[i+1, j], distance[i, j-1], distance[i, j+1]):
                    local_maxima[i, j] = True
        return local_maxima

    def label_regions(local_maxima):
        labels = np.zeros_like(local_maxima, dtype=int)
        current_label = 1
        for i in range(local_maxima.shape[0]):
            for j in range(local_maxima.shape[1]):
                if local_maxima[i, j] and labels[i, j] == 0:
                    flood_fill(local_maxima, labels, i, j, current_label)
                    current_label += 1
        return labels

    def flood_fill(image, labels, x, y, label):
        to_fill = [(x, y)]
        while to_fill:
            i, j = to_fill.pop()
            if labels[i, j] == 0 and image[i, j]:
                labels[i, j] = label
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                        to_fill.append((ni, nj))

    # Convert image to binary format (0 and 1)
    binary_image = (image > 0).astype(int)
    
    distance = distance_transform_edt(binary_image)
    local_maxima = find_local_maxima(distance)
    markers = label_regions(local_maxima)
    labeled_image = np.zeros_like(image, dtype=int)

    sorted_indices = np.argsort(-distance, axis=None)
    for idx in sorted_indices:
        i, j = np.unravel_index(idx, distance.shape)
        if binary_image[i, j] and labeled_image[i, j] == 0:
            neighbor_labels = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                    if labeled_image[ni, nj] > 0:
                        neighbor_labels.append(labeled_image[ni, nj])
            if neighbor_labels:
                labeled_image[i, j] = max(set(neighbor_labels), key=neighbor_labels.count)
            else:
                labeled_image[i, j] = markers[i, j]

    return labeled_image