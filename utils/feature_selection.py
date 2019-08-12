##========================================================================
##-------------------------Hugo, Allyson, Roberto, Rizan-----------------
##========================================================================

#Import:
import nn
from helper import read_csv

from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import time

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2

from keras.utils import to_categorical
from keras.callbacks import History, EarlyStopping
from keras.models import load_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":

        #Read data:
        X, y, features_name = read_csv('data.csv')

        #Preprocessing:
        X = minmax_scale(X)
        y = to_categorical(y)
        
        #Get Information:
        selector = SelectKBest(chi2).fit(X, y)
        idx = np.argsort(selector.scores_)
        print(idx)

        print("Features: ")
        for i in range(11):
            print(i, features_name[i])

        print("\nScores: ")
        for a,b in zip(features_name[idx], selector.scores_[idx]):
            print(a, b)
        print("--------------------------------")

        #Evaluate based on accuracy:
        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        for i in range(1,11):

            #Create model
            model = nn.create_mlp(i, (100,10), 4)

            #Select Features
            X_new = SelectKBest(chi2, k=i).fit_transform(X, y)
            
            #Split
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=20000, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=60000, random_state=1)

            #Train
            model.fit(X_train, y_train, epochs=500, batch_size=100, validation_data=(X_val, y_val), callbacks=[earlyStopping], verbose=0)

            #Calculate Accuracy
            acc = round(model.evaluate(X_test, y_test, verbose=0)[1], 4)

            #Print Accuracy
            print("k = " + str(i), "Accuracy: ", acc)
            print()
