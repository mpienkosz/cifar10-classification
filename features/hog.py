from skimage import color, feature
from tqdm import tqdm
import numpy as np


class HogFeatureExtractor:

    # A class responsible for extracting Histogram of Oriented Gradients (HoG) features from pre-trained Inception-v3 network

    def extract_features(self, data):
        data_gray = [color.rgb2gray(instance) for instance in data]
        data_hog = [feature.hog(instance, cells_per_block=(2, 2)) for instance in tqdm(data_gray)]
        return np.array(data_hog)