import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename): 

    with open(filename, newline="") as csvfile:
        data = csv.reader(csvfile)
        data = np.array(list(data))

    features_name = data[0, :-1]
    X = data[1:, :-1].astype(np.float)
    y = data[1:, -1].astype(np.int)

    return X, y, features_name

def plot_clustering(X_red, labels, y, title, savepath):
    # From https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html
    # Author: Gael Varoquaux
    # Distributed under BSD license
    #
    # X_red must be a numpy array containing the features
    # input data, reduced to 2 dimensions
    #
    # labels must be a numpy array containing the labels of each of the
    # elements of X_red, in the same order
    #
    # title is the title you want to give to the figure
    #
    # savepath is the name of the file where the figure must be saved
    #
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(9, 6), dpi=160)
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                color=plt.cm.nipy_spectral(labels[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(savepath)
    plt.close()
