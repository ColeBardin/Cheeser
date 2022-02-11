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
def get_train_test (X, y, f_tr, names):
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
    # Make list of all filenames for testing data 
    names_te = [names[i] for i in range(n) if i not in i_tr]
    # return training and testing
    return X_tr , X_te , y_tr , y_te , names_te

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
            print(f"Learning about {subdir}...")
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
    # Base name of .pkl file
    base_name = 'cheese_or_not'
    # Desired image width after resize
    width = 100

    print("Cheesing...")
    # Check for number of arguments passed
    if len(argv) == 2:
        # Validate init flag
        if argv[1] == 'init':
            # Create path to data directories
            data_path = os.path.join("data")
            # Subdirectories of data to include
            include = {'Cheese', 'NotCheese'}
            print("Reading and resizing all the data images")
            # Check if the .pkl file exists already
            if os.path.isfile(f'{base_name}_{width}x{width}px.pkl'):
                # If it does, delete it
                os.remove(f'{base_name}_{width}x{width}px.pkl')
            # Make new .pkl file with the data path, filename, resize width and included subdirectories
            resize_all(src=data_path, pklname=base_name, width=width, include=include)
        # If more another arg is passed but it is not init flag
        else:
            # Print usgae of program
            print('Usage: cheeser.py [init]')
            # End program with status -1
            return -1
 
    # Validate if .pkl file exists
    if os.path.isfile(f'{base_name}_{width}x{width}px.pkl'):
        print(f"Loading {base_name}_{width}x{width}px.pkl...")
        # Load the data from the .pkl file
        data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    # If the .pkl file doesn't exist
    else:
        # Error
        print("Cannot find .pkl data file")
        # End program with status -1
        return -1
 
    # Iterate over each data entry
    for index in range(len(data['label'])):
        # Fix truncated label due to np.unique
        data['label'][index] = data['label'][index] + 'eese'

    # Print data informatgion
    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']))
 
    # Make counter dict of data
    Counter(data['label'])
    
    # Use np.unique to get all unique values in the list of labels
    labels = np.unique(data['label']) 

    # Set up the matplotlib figure and axes, based on the number of labels to display training data examples
    fig, axes = plt.subplots(1, len(labels))
    fig.suptitle("Examples from training data")
    fig.set_size_inches(14,4)
    fig.tight_layout()
 
    # Make a plot for every label (equipment) type. The index method returns the 
    # Index of the first item corresponding to its search string, label in this case
    for ax, label in zip(axes, labels):
        # Create empty list of choices
        choices = []
        # Sort data by current label
        for index in range(len(data['label'])):
            if data['label'][index] == label:
                choices.append(index)
        # Choose a random index from the list of chocies
        idx = choices[randint(0, len(choices)-1)]
        # Display the image
        ax.imshow(data['data'][idx])
        # Format the plot
        ax.axis('off')
        filename = 'filename'
        ax.set_title(f'{data[filename][idx]}')

    # Turn the dictionary data into np arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    names = np.array(data['filename'])

    # Split all data into training and testing based on desired ratio
    print("Splitting training and testing data")
    X_train, X_test, y_train, y_test, names_te = get_train_test(X=X, y=y, f_tr=0.9, names=names)

    # Create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()
 
    # Grayify training data
    print("Grayifying training data")
    X_train_gray = grayify.fit_transform(X_train)
    # HOGify the grayified training data
    print("HOGifying training data")
    X_train_hog = hogify.fit_transform(X_train_gray)
    # Transform the HOGified training data
    print("Fitting the transfromed train data")
    X_train_prepared = scalify.fit_transform(X_train_hog)
 
    # Output the shape of the prepared data
    print(X_train_prepared.shape)

    # Make the SDG Classifier instance
    print("Training the SDG Classifier")
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    # Train the SDG Classifier
    sgd_clf.fit(X_train_prepared, y_train)

    # Grayify tesing data
    print("Grayifing the testing data")
    X_test_gray = grayify.transform(X_test)
    # HOGify the grayified testing data
    print("HOGifying the testing data")
    X_test_hog = hogify.transform(X_test_gray)
    # Transform the HOGified testing data
    print("Fitting the transofmred testing data")
    X_test_prepared = scalify.transform(X_test_hog)

    # Test the SDG Classifier with the prepared testing data
    print("Estimating the Cheesiness of the testing data")
    y_pred = sgd_clf.predict(X_test_prepared)
    # Print the first 25 results of the testing predictions
    print(np.array(y_pred == y_test)[:25])

    # Generate the confusion matrix
    print("Generating confusion matrix")
    cmx = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cmx)

    # Empty list to hold incorrect indexes
    incorrect_idx = []
    # Iterate through predicted results
    for index in range(len(y_pred)):
        # If the prediction is wrong
        if y_pred[index] != y_test[index]:
            # Save the index of incorrect prediction
            incorrect_idx.append(index)

    # Output results of the prediction
    print('')
    print(f"Number of incorrect predictions: {len(incorrect_idx)} out of {len(y_pred)} examples")
    print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

    # If there are less than 6 incorrect answers
    if len(incorrect_idx) <= 6:
        # Only use those incorrect answers
        num = len(incorrect_idx)
        # The chosen indexes should be all the incorrect predictions
        rand_indices = incorrect_idx
    # If there are more than 6 incorrect predictions
    else:
        # Only allow for 6 to be displayed
        num = 6
        # Choose 6 random indices from list of incorrect indices
        rand_indices = np.random.choice(incorrect_idx, size=num, replace=False)

    print(f"Displaying {num} incorrect guesses")
    # Set up the matplotlib figure and axes, based on the number of labels
    fig2, axes2 = plt.subplots(1, num)
    fig2.suptitle(f"{num} incorrect predictions from testing data")
    fig2.set_size_inches(14,4)
    fig2.tight_layout()

    # Iterate over each axis and index
    for ax, idx in zip(axes2, rand_indices):
        # Display the image
        ax.imshow(X_test[idx])
        # Format the graph
        #ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{names_te[idx]}")
        ax.set_title(f"This is {y_pred[idx]}")
    # Show the plot and wait for user to close it
    plt.show()

if __name__ == '__main__':
    main()
