import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# A helper function for plotting dataset of 2-D features (different collors for different classes)
def plot_features(X_proj, Y_train):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'gray', 'orange']
    Y_colors = [colors[val] for val in Y_train]
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=Y_colors, s=5)
    plt.savefig('resources/features.png')

# A helper function for plotting sample images from dataset
def plot_images(images, labels, num_samples):
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axes = plt.subplots(num_samples,num_samples)
    for row in range(num_samples):
        images_for_label = [img for img, label in zip(images, labels) if label == row]
        random_indices = np.random.choice(range(len(images_for_label)), num_samples, replace=False)
        for col, random_idx in enumerate(random_indices):
            axes[row][col].imshow(images_for_label[random_idx])
            axes[row][col].get_xaxis().set_ticks([])
            axes[row][col].get_yaxis().set_ticks([])
        axes[row][0].set_ylabel(cifar_classes[row], labelpad=50,  rotation=0)
    #plt.subplots_adjust(left=0.1, right=0.2)
    plt.savefig('resources/cifar_sample', bbox_inches='tight')