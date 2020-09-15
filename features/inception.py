import numpy as np
import scipy.misc
import cv2
from keras.applications import inception_v3
from utils.configuration import INPUT_IMG_SIZE
from tqdm import tqdm
from math import ceil

class InceptionFeatureExtractor:

    # A class responsible for extracting features from pre-trained Inception-v3 network

    def __init__(self):
        self.__model = inception_v3.InceptionV3(input_shape=(INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3),
                           weights='imagenet',
                           include_top=False,
                            pooling='avg')

    # A method for extracting Inception-v3 features based on input data
    def extract_features(self, input, batch_size=256):
        processing_steps = ceil(len(input)/batch_size)
        features = self.__model.predict_generator(generator = self.input_processing_generator(input, batch_size),
                                                  steps=processing_steps,
                                                  verbose=1)
        return features.reshape(features.shape[0], -1)

    # A method that resizes and preprocesses batch of input data for Inception-v3 network
    def resize_and_preprocess_batch(self, input_batch):
        processed_batch = inception_v3.preprocess_input(input_batch.astype(np.float32))
        processed_batch = [cv2.resize(image, (INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                           interpolation=cv2.INTER_CUBIC)
                        for image in processed_batch]
        return np.array(processed_batch)

    # A generator for batches of input data
    def input_processing_generator(self, input, batch_size=50):
        while True:
            for start_id in range(0, len(input), batch_size):
                input_batch = input[start_id:start_id + batch_size]
                yield self.resize_and_preprocess_batch(input_batch)

    # A generator for batches of input-target data
    def input_target_processing_generator(self, inputs, targets, batch_size=50, n_epoch=1):
        while True:
            for start_id in range(0, len(inputs), batch_size):
                input_batch = inputs[start_id:start_id + batch_size]
                target_batch = targets[start_id:start_id + batch_size]
                yield (self.resize_and_preprocess_batch(input_batch), target_batch)

