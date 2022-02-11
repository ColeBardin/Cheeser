import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import joblib 
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
from random import randint
import random
from sys import argv
from transformer_classes import *

#Split dataset into training vs testing data. usually split 50/50 randomly is sufficient
def get_train_test (X, y, f_tr):
    # get number of instances
    n = X.shape[0]
    # set number of images for training , testing
    n_tr = int(f_tr * n)
    n_te = n - n_tr
    # pick indices for training
    i_tr = np.random.choice(n, n_tr , replace=False)
    # split X_lst into training and testing
    X_tr = [X[i] for i in range(n) if i in i_tr]
    X_te = [X[i] for i in range(n) if i not in i_tr]
    # split y_lst into training and testing
    y_tr = [y[i] for i in range(n) if i in i_tr]
    y_te = [y[i] for i in range(n) if i not in i_tr]
    # return training and testing
    return X_tr , X_te , y_tr , y_te

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)

def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
         
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
    plt.show()

def main():
    base_name = 'cheese_or_not'
    width = 100

    if len(argv) == 2:
        if argv[1] == 'init':
            data_path = os.path.join("data")
            include = {'Cheese', 'NotCheese'}
            resize_all(src=data_path, pklname=base_name, width=width, include=include)
        else:
            print('Usage: cheeser.py [init]')
            return -1
 
    if os.path.isfile(f'{base_name}_{width}x{width}px.pkl'):
        data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    else:
        print("Cannot find .pkl data file")
        return -1
 
    for index in range(len(data['label'])):
        data['label'][index] = data['label'][index] + 'eese'

    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))
 
    Counter(data['label'])
    
    # use np.unique to get all unique values in the list of labels
    labels = np.unique(data['label']) 

    # set up the matplotlib figure and axes, based on the number of labels
    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15,4)
    fig.tight_layout()
 
    # make a plot for every label (equipment) type. The index method returns the 
    # index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        choices = []
        for index in range(len(data['label'])):
            if data['label'][index] == label:
                choices.append(index)

        idx = choices[randint(0, len(choices))]
     
        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)

    X = np.array(data['data'])
    y = np.array(data['label'])

    X_train, X_test, y_train, y_test = get_train_test(X, y, 0.9)

    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()
 
    # call fit_transform on each transform converting X_train step by step
    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)
 
    print(X_train_prepared.shape)

    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    y_pred = sgd_clf.predict(X_test_prepared)
    print(np.array(y_pred == y_test)[:25])
    print('')
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

    cmx = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cmx)

    # set up the matplotlib figure and axes, based on the number of labels
    fig, axes = plt.subplots(1, 6)
    fig.set_size_inches(15,4)
    fig.tight_layout()

    rand_indices = np.random.choice(range(len(y_pred)), size=6, replace=False)

    for ax, idx in zip(axes, rand_indices):
        ax.imshow(X_test[idx])
        ax.axis('off')
        ax.set_title("This is {}".format(y_pred[idx]))
    plt.show()

if __name__ == '__main__':
    main()
