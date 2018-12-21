#Import
from helper import read_csv, plot_clustering

import numpy as np
import csv

from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn import decomposition

import matplotlib.pyplot as plt


if __name__ == "__main__":

    #Read data:
    X_data, y, features_name = read_csv('data.csv')

    
    importance_order = [7, 0, 10, 8, 2, 9, 4, 5, 6, 1, 3]
    for i in range(2, 12):

        #Select Features and number of points: 
        X = X_data[:5000,  importance_order[:i]]

        #Number of points to train and to plot:
        nb_train = 100000
        nb_test = 5000

        #Preprocessing:
        X = scale(X)

        #Method:
        #tsne = TSNE(n_components=2,learning_rate=200.0, n_iter=1500, early_exaggeration=5.0, perplexity=100, random_state=0,verbose=2)
        pca = decomposition.PCA(n_components=2)

        #X Embedded
        #X_embedded = tsne.fit_transform(X)
        X_embedded = pca.fit(X[:nb_train]).transform(X[:nb_test])

        #Plot:
        plot_clustering(X_embedded[:nb_test], y[:nb_test], y, "", "images/pca_" + str(i) + ".png")