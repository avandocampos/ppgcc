import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA

def extract_hog_features(images):
    hog_features = []
    for image in images:
        features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        hog_features.append(features)
    return np.array(hog_features)


def extract_lbp_features(images):
    lbp_features = []
    for image in images:
        lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)
    return np.array(lbp_features)


def extract_pca_features(images, n_components=50):
    pca = PCA(n_components=n_components)
    images_flat = images.reshape(images.shape[0], -1)
    pca_features = pca.fit_transform(images_flat)
    return pca_features

