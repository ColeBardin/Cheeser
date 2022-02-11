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
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn import svm
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

# Method to resize all of the images and save a .pkl file of the data
def resize_all(src, pklname, include, width=150, height=None):
    # Parameters:
    # src: str
        # path to data
    # pklname: str
        # path to output file
    # width: int
        # target width of the image in pixels
    # include: set[str]
        # set containing str

    # If specified height is given, use that else use the default width     
    height = height if height is not None else width
    # Create empty dictionary
    data = dict()
    # Set the description
    data['description'] = 'resized ({0}x{1}) pixel images in rgb'.format(int(width), int(height))
    # Initialize empty values for the keys label, filename and data
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
    # Create .pkl file name
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # Read all subdirectories from the source
    for subdir in os.listdir(src):
        # If the sundirectory is instructed to be included
        if subdir in include:
            print(f"Learning about {subdir}...")
            # Create the path to the subdirectory
            current_path = os.path.join(src, subdir)
            # Iterate over each file in the subdirectory
            for file in os.listdir(current_path):
                # Only read from .jpg and .png files
                if file[-3:] in {'jpg', 'png'}:
                    # Read the image
                    im = imread(os.path.join(current_path, file))
                    # Resize it with the specified format
                    im = resize(im, (width, height)) #[:,:,::-1]
                    # Add entry of label to subdirectory
                    data['label'].append(subdir[:-4])
                    # Add filename entry
                    data['filename'].append(file)
                    # Append the image data to the dictionary
                    data['data'].append(im)
        # Create a .pkl file to store image data for later reference
        joblib.dump(data, pklname)

# Method to plot the confusion matrices
def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    # Create a numpy array with the corresponding respecive percentages of the cmx cells
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    # Make a copy of this array
    cmx_zero_diag = cmx_norm.copy()
    # Fill it's diagonals with 0
    np.fill_diagonal(cmx_zero_diag, 0)
 
    # Create figure with 3 subplots
    fig, ax = plt.subplots(ncols=3)
    # Set figure title
    fig.suptitle("Confusion Matrices")
    # Set the figure size
    fig.set_size_inches(12, 4)
    # Add axis tick markers based on amount of incoming data
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
         
    # Count confusion matrix
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    # Set subplot title
    ax[0].set_title('Count')
    # Set subplot X label
    ax[0].set_xlabel("Predicted")
    # Set subplot Y label
    ax[0].set_ylabel("True Label")
    # Set X axis tick markers
    ax[0].set_xticklabels(['Cheese','NotCheese',''])
    # Set Y axis tick markers
    ax[0].set_yticklabels(labels=['Cheese','NotCheese',''], rotation=45)

    # Percentage confusion matrix
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    # Set subplot title
    ax[1].set_title('Percentage')
    # Set X label
    ax[1].set_xlabel("Predicted")
    # Set Y label
    ax[1].set_ylabel("True Label")
    # Set X axis tick markers
    ax[1].set_xticklabels(['Cheese','NotCheese',''])
    # Set Y axis tick markers
    ax[1].set_yticklabels(labels=['Cheese','NotCheese',''], rotation=45)

    # % and 0 Diagonal confusion matrix
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    # Set the subplot title
    ax[2].set_title('% and 0 diagonal')
    # Set X label
    ax[2].set_xlabel("Predicted")
    # Set Y label
    ax[2].set_ylabel("True Label")
    # Set X tick markers
    ax[2].set_xticklabels(['Cheese','NotCheese',''])
    # Set Y tick markers
    ax[2].set_yticklabels(labels=['Cheese','NotCheese',''], rotation=45)
 
    # Create dividers between the subplots
    dividers = [make_axes_locatable(a) for a in ax]
    # Size and fit the axis with the padding
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    # Add color bars to each of the axis with their ranges
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    # Enable tight layout
    fig.tight_layout()
    # Uncomment plt.show() to have program wait until confusion matrix and data examples are closed
    #plt.show()

def main():
    # Base name of .pkl file
    base_name = 'cheese_or_not'
    # Desired image width after resize
    width = 100

    print("Cheesing...\n")
    # Check for number of arguments passed
    if len(argv) == 2:
        # Validate init flag
        if argv[1] == 'init':
            # Create path to data directories
            data_path = os.path.join("data")
            # Subdirectories of data to include
            include = {'Cheese', 'NotCheese'}
            print("Reading and resizing all the data images...")
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
        print(f"Loading {base_name}_{width}x{width}px.pkl...\n")
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

    # Set to True to print out examples of the data images
    print_examples = False

    # If the data examples should be printed
    if print_examples == True:
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
    print("\nSplitting training and testing data")
    X_train, X_test, y_train, y_test, names_te = get_train_test(X=X, y=y, f_tr=0.9, names=names)

    # Set up the HOG pipeline for optimized search
    print("\nCreating the HOG pipeline to optimze search")
    HOG_pipeline = Pipeline([
        # Transformers
        ('grayify', RGB2GrayTransformer()),
        ('hogify', HogTransformer(
            pixels_per_cell=(14, 14), 
            cells_per_block=(2, 2), 
            orientations=9, 
            block_norm='L2-Hys')
        ),
        ('scalify', StandardScaler()),
        ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    ])

    # Generate a Classifier with only the hog pipeline
    clf = HOG_pipeline.fit(X_train, y_train)

    # Generate a prediction with only the HOG Pipeline
    y_pred_clf = clf.predict(X_test)
    # Calculate accuracy of Pipeline transform fit
    clf_accuracy = 100*np.sum(y_pred_clf == y_test)/len(y_test)
 
    # Set up grid parameters
    param_grid = [
    {
        'hogify__orientations': [8, 9],
        'hogify__cells_per_block': [(2, 2), (3, 3)],
        'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)]
    },
    {
        'hogify__orientations': [8],
         'hogify__cells_per_block': [(3, 3)],
         'hogify__pixels_per_cell': [(8, 8)],
         'classify': [
             SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
             svm.SVC(kernel='linear')
         ]
    }
    ]

    # Create a grid search with the HOG pipeline
    print("\nCreating Grid Search framework")
    grid_search = GridSearchCV(HOG_pipeline, 
                           param_grid, 
                           cv=3,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=1,
                           return_train_score=True)
 
    # Train the grid search to find the best descriptors
    print("Training the grid search\n")
    grid_res = grid_search.fit(X_train, y_train)

    # Print description of best performing object
    #print(grid_res.best_estimator_)

    # Use the grid search results to predict the dest data
    print("\nUsing best performing descriptors of Grid Search to predict test data")
    y_pred_grid = grid_res.predict(X_test)
    # Calculate accuracy of grid search
    grid_accuracy = 100*np.sum(y_pred_grid == y_test)/len(y_test)
    # Print out the accuracy of the CLF Pipeline fit
    print(f"\nCLF is {clf_accuracy}% accurate")
    # Print out the accuracy of the grid search
    print(f"Grid search is {grid_accuracy}% accurate\n")

    # Compare accuracy of Grid search vs Pipeline transform fit
    if grid_accuracy > clf_accuracy:
        # Grid search is more accurate
        y_pred = y_pred_grid
    else:
        # Pipeline is more accurate
        y_pred = y_pred_clf

    # If the model has a greater accuracy than 90%
    if 100*np.sum(y_pred == y_test)/len(y_test) > 90:
        # Save the model to disk
        joblib.dump(grid_res, 'hog_sgd_model.pkl')

    # Generate the confusion matrix
    print("Generating confusion matrix")
    cmx = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrices
    plot_confusion_matrix(cmx)

    # Empty list to hold incorrect indexes
    incorrect_idx = []
    # Iterate through predicted results
    for index in range(len(y_pred)):
        # If the prediction is wrong
        if y_pred[index] != y_test[index]:
            # Save the index of incorrect prediction
            incorrect_idx.append(index)

    # Print the results table
    print("\nFilename\tTrue Label\tPrediction")
    for index in range(len(y_pred)):
        print(f"{names_te[index]}\t{y_test[index]}\t{y_pred[index]}")

    # Output results of the prediction
    print(f"\nNumber of incorrect predictions: {len(incorrect_idx)} out of {len(y_pred)} examples")
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

    # Set up the matplotlib figure and axes, based on the number of labels
    fig2, axes2 = plt.subplots(1, num)
    fig2.suptitle(f"{num} incorrect predictions from testing data")
    fig2.set_size_inches(14,4)
    fig2.tight_layout()

    # Iterate over each axis and index
    for ax, idx in zip(axes2, rand_indices):
        # Display the image
        ax.imshow(X_test[idx])
        # Format the graph title
        ax.set_title(f"This is {y_pred[idx]}")
        # Turn off axis tick markers
        ax.set_xticks([])
        ax.set_yticks([])
        # Set the X Label to the filename
        ax.set_xlabel(f"{names_te[idx]}")
    # Show the plot and wait for user to close it
    plt.show()

if __name__ == '__main__':
    main()
