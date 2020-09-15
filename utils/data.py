import random
import numpy as np

# A function that returns random sample of the data
def sample_dataset(inputs, labels, sample_percent):
    data_size = len(inputs)
    sample_idx = random.sample(range(data_size), int(sample_percent*data_size))
    sample_inputs = [sample for id, sample in enumerate(inputs) if id in sample_idx]
    sample_labels = [sample for id, sample in enumerate(labels) if id in sample_idx]
    return np.array(sample_inputs), np.array(sample_labels)

# A function that splits data to two (e.g. dev/test) splits
def split_data(inputs, labels, dev_split_fraction):
    data_size = len(inputs)
    dev_size = int(dev_split_fraction * data_size)
    data_dev = inputs[:dev_size]
    data_train = inputs[dev_size:]
    labels_dev = labels[:dev_size]
    labels_train = labels[dev_size:]
    return (data_train, labels_train), (data_dev, labels_dev)