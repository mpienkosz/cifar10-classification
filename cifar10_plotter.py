from utils.visualization import plot_images

from keras.datasets import cifar10

print('Loading cifar10..')
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

print('Plotting..')
plot_images(X_train, Y_train, 10)
