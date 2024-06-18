import numpy as np

def region_growing(image, seed, threshold):
    rows, cols = image.shape
    segmented_image = np.zeros_like(image)
    region_mean = float(image[seed])
    segmented_image[seed] = 255
    to_process = [seed]
    processed = set()
    processed.add(seed)

    while to_process:
        x, y = to_process.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in processed:
                if abs(int(image[nx, ny]) - int(region_mean)) < threshold:
                    segmented_image[nx, ny] = 255
                    # Update the region mean based on new pixel
                    region_mean = (region_mean * len(processed) + image[nx, ny]) / (len(processed) + 1)
                    to_process.append((nx, ny))
                    processed.add((nx, ny))

    return segmented_image
