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

from keras.utils import to_categorical
from keras.callbacks import History
from keras.models import load_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def table_plot(data, col_labels, row_labels, title1, idx, accs):
    cols = np.append(np.append(idx, col_labels), 'Accuracy')
    data = np.array(data).astype(str)
    accs = np.array([accs])
    data = np.concatenate( (data, accs.T), axis=1)
    row_labels = np.array([row_labels])
    cells = np.concatenate( (row_labels.T, data), axis=1)
    x = PrettyTable(cols.tolist())
    for row in cells:
        x.add_row(row.tolist())
    print(x.get_string(title=title1))

def print_database_information(y, y_train, y_val, y_test):

    _, counts_all   = np.unique(y, return_counts=True, axis=0)
    _, counts_train = np.unique(y_train, return_counts=True, axis=0)
    _, counts_val   = np.unique(y_val, return_counts=True, axis=0)
    _, counts_test  = np.unique(y_test, return_counts=True, axis=0)

    print("\n-----------------------------------")
    print("All Dataset:")
    print("-----------------------------------")
    for i in range(4):
        print("class", i, ': ', counts_all[i], '|', round(counts_all[i]/len(y)*100, 2), '%')
    print("-----------------------------------\n")

    print("-----------------------------------")
    print("Train Dataset:")
    print("-----------------------------------")
    for i in range(4):
        print("class", i, ': ', counts_train[i], '|', round(counts_train[i]/len(y_train)*100, 2), '%')
    print("-----------------------------------\n")

    print("-----------------------------------")
    print("Validation Dataset:")
    print("-----------------------------------")
    for i in range(4):
        print("class", i, ': ', counts_val[i], '|', round(counts_val[i]/len(y_val)*100, 2), '%')
    print("-----------------------------------\n")

    print("-----------------------------------")
    print("Test Dataset:")
    print("-----------------------------------")
    for i in range(4):
        print("class", i, ': ', counts_test[i], '|', round(counts_test[i]/len(y_test)*100, 2), '%')
    print("-----------------------------------\n")


def execute_paper():
        #Train Neural Networks and Plot the image
        fig, axs = plt.subplots(nrows=1, ncols=2)
        titles  =['1:(5)', '1:(10)', '1:(100)', '2:(100x10)']
        colors = ['r', 'g', 'b', 'k']
        
        hidden_layers=[(5,), (10,), (100,), (100,10,)]
        neural_networks = [nn.create_mlp(11, hidden_layers[i], 4) for i in range(4)]

        earlyStopping = nn.MyEarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
        total_time = [0 for i in range(4)]

        for i in range(4):
            print(titles[i] + ' Start')
            
            curr_time = time.time()
            history = neural_networks[i].fit(X_train, y_train, epochs=500, batch_size=100, validation_data=(X_val, y_val), callbacks=[earlyStopping], verbose=0)

            total_time[i] = time.time() - curr_time
            optimal_point = earlyStopping.stopped_epoch

            #Plot:
            y_min = min(history.history["val_acc"])
            y_max = history.history["val_acc"][optimal_point]

            axs[0].plot(range(len(history.history["val_acc"])), history.history["val_acc"], color=colors[i], label=titles[i])
            axs[0].vlines(x=optimal_point, ymin=0.6, ymax=y_max, color='k', linestyle='--')
            legend = str(optimal_point) + ", " + str(round(history.history["val_acc"][-1]*100, 1)) + "%"
            axs[0].text(optimal_point+10, y_min+0.1, legend)#, ha='center', va='center',rotation='vertical')

            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Classification Accuracy")

            print(titles[i]+' Done!')
        axs[0].legend()
        
        def secs(x, pos):
            'The two args are the value and tick position'
            return '%1.1fs' % (x)
        formatter = FuncFormatter(secs)
        axs[1].yaxis.set_major_formatter(formatter)
        axs[1].bar(titles, total_time)

        plt.show()

def print_confusion_matrix():

    tab_idx = ['a', 'b', 'c', 'd']
    classes = ['DP-BPSK', 'DP-QPSK', 'DP-16QAM', 'DP-64QAM']
    titles  =['1:(5)', '1:(10)', '1:(100)', '2:(100x10)']

    models = ["allFeatures_11-[5]-4", "allFeatures_11-[10]-4", "allFeatures_11-[100,10]-4", "allFeatures_11-[100]-4"]
        
    nothot_y_test = np.argmax(y_test, axis=1)
    for i in range(len(tab_idx)):

        clf = load_model("trained_models/" + models[i] + ".h5")

        y_pred = np.argmax(clf.predict(X_test), axis=1)
        acc = np.sum(y_pred==nothot_y_test)/len(nothot_y_test)

        accs = []
        for c in range(len(classes)):
            idx = nothot_y_test==c
            curr_acc = round(clf.evaluate(X_test[idx], y_test[idx], verbose=0)[1], 4)
            accs.append(curr_acc)

        cnf_matrix = confusion_matrix(nothot_y_test,y_pred)
        table_plot(cnf_matrix, classes, classes, titles[i], tab_idx[i]+')'+titles[i]+' Acc='+str(round(acc, 4)) , accs)
        

if __name__ == "__main__":

        #Read data:
        X, y, features_name = read_csv('data.csv')

        #Preprocessing:
        X = scale(X)
        y = to_categorical(y)

        #Split the dataset:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20000, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=60000, random_state=1)

        #Database Information:
        #print_database_information(y, y_train, y_val, y_test)

        #Train Neural Networks and Plot the image
        #execute_paper()
        
        #Print table
        print_confusion_matrix()

