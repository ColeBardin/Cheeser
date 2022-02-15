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
def get_train_test (X, y, f_tr, names, test=False, indir_data=None):
    # get number of instances
    n = X.shape[0]
    # set number of images for training , testing
    n_tr = int(f_tr * n)
    n_te = n - n_tr
    # pick indices for training
    i_tr = np.random.choice(n, n_tr , replace=False)
    # split X_lst into training and testing
    X_tr = [X[i] for i in range(n) if i in i_tr]
    # If not testing indir data
    if test == False:
        # Set X testing data
        X_te = [X[i] for i in range(n) if i not in i_tr]
        # Set Y testing data
        y_te = [y[i] for i in range(n) if i not in i_tr]
    # If indir data is passed correctly
    elif indir_data is not None:
        # Set X testing data
        X_te = indir_data[0]
        # Set Y testing data
        y_te = indir_data[1]
    # split y_lst into training and testing
    y_tr = [y[i] for i in range(n) if i in i_tr]
    # Make list of all filenames for testing data 
    names_te = [names[i] for i in range(n) if i not in i_tr]
    # return training and testing
    return X_tr , X_te , y_tr , y_te , names_te

# Method to resize all of the images from the indir testing diredtory
def resize_indir(src, width=150, height=None):
    # If specified height is given, use that else use the default width     
    height = height if height is not None else width
    # Create empty dictionary
    testing = dict()
    # Set the description
    testing['description'] = 'resized ({0}x{1}) testing data from indir\\ in rgb'.format(int(width), int(height))
    # Initialize empty values for the keys label, filename and data
    testing['label'] = []
    testing['filename'] = []
    testing['data'] = []   
 
    # Iterate over each file in indir
    for file in os.listdir(src):
        # Only read from .jpg and .png files
        if file[-3:] in {'jpg', 'png'}:
            # Read the image
            im = imread(os.path.join(src, file))
            # Resize it with the specified format
            im = resize(im, (width, height)) #[:,:,::-1]
            # Add entry of label to subdirectory
            testing['label'].append("Unknown")
            # Add filename entry
            testing['filename'].append(file)
            # Append the image data to the dictionary
            testing['data'].append(im) 
        # Return the list of testing data
    return testing

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
                    data['label'].append(subdir)
                    # Add filename entry
                    data['filename'].append(file)
                    # Append the image data to the dictionary
                    data['data'].append(im)
    # Print a newline
    print('')
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
    fig_cmx, axes_cmx = plt.subplots(ncols=3)
    # Set figure title
    fig_cmx.suptitle("Confusion Matrices")
    # Set the figure size
    fig_cmx.set_size_inches(12, 4)
    # Add axis tick markers based on amount of incoming data
    [a.set_xticks(range(len(cmx)+1)) for a in axes_cmx]
    [a.set_yticks(range(len(cmx)+1)) for a in axes_cmx]
         
    # Count confusion matrix
    im1 = axes_cmx[0].imshow(cmx, vmax=vmax1)
    # Set subplot title
    axes_cmx[0].set_title('Count')
    # Set subplot X label
    axes_cmx[0].set_xlabel("Predicted")
    # Set subplot Y label
    axes_cmx[0].set_ylabel("True Label")
    # Set X axis tick markers
    axes_cmx[0].set_xticklabels(['Cheese','NotCheese',''])
    # Set Y axis tick markers
    axes_cmx[0].set_yticklabels(labels=['Cheese','NotCheese',''], rotation=45)

    # Percentage confusion matrix
    im2 = axes_cmx[1].imshow(cmx_norm, vmax=vmax2)
    # Set subplot title
    axes_cmx[1].set_title('Percentage')
    # Set X label
    axes_cmx[1].set_xlabel("Predicted")
    # Set Y label
    axes_cmx[1].set_ylabel("True Label")
    # Set X axis tick markers
    axes_cmx[1].set_xticklabels(['Cheese','NotCheese',''])
    # Set Y axis tick markers
    axes_cmx[1].set_yticklabels(labels=['Cheese','NotCheese',''], rotation=45)

    # % and 0 Diagonal confusion matrix
    im3 = axes_cmx[2].imshow(cmx_zero_diag, vmax=vmax3)
    # Set the subplot title
    axes_cmx[2].set_title('% and 0 diagonal')
    # Set X label
    axes_cmx[2].set_xlabel("Predicted")
    # Set Y label
    axes_cmx[2].set_ylabel("True Label")
    # Set X tick markers
    axes_cmx[2].set_xticklabels(['Cheese','NotCheese',''])
    # Set Y tick markers
    axes_cmx[2].set_yticklabels(labels=['Cheese','NotCheese',''], rotation=45)
 
    # Create dividers between the subplots
    dividers = [make_axes_locatable(a) for a in axes_cmx]
    # Size and fit the axis with the padding
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    # Add color bars to each of the axis with their ranges
    fig_cmx.colorbar(im1, cax=cax1)
    fig_cmx.colorbar(im2, cax=cax2)
    fig_cmx.colorbar(im3, cax=cax3)
    # Enable tight layout
    fig_cmx.tight_layout()
    # Uncomment plt.show() to have program wait until confusion matrix and data examples are closed
    #plt.show()
    axes_cmx[0].set_xlabel("Predicted")
    axes_cmx[0].set_ylabel("True Label")

# Method to get target answer from the question
def get_answer(message, targets):
    # Prompt user
    answer = input(message)
    # Check if response is in target answers
    for target in targets:
        # If there is a match
        if answer.lower() == target.lower():
            # Return the answer
            return answer
    # If there is no match implement recursion
    return get_answer(message, targets)

# Method to create GridSearch and use it to get grid_res
def get_grid_res(X_train, y_train):
    # Get the HOG Pipeline
    print("Creating the HOG pipeline to optimze search\n")
    HOG_pipeline = get_HOG_pipeline()
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
    }]
    # Create a grid search with the HOG pipeline
    print("Creating Grid Search framework\n")
    grid_search = GridSearchCV(HOG_pipeline, 
                    param_grid, 
                    cv=10,
                    n_jobs=-1,
                    scoring='accuracy',
                    verbose=1,
                    error_score='raise',
                    return_train_score=True)
    # Create the grid search method
    return grid_search.fit(X_train, y_train)

# Method to generate and get a HOG pipeline
def get_HOG_pipeline():
    # Set up the HOG pipeline for optimized search
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
    # Return the final pipeline
    return HOG_pipeline

# Main program function
def main():
    print("\nCheesing...\n")
    # Base name of .pkl file
    base_name = 'cheese_or_not'
    # Desired image width after resize
    width = 100
    # Define usage string
    usage = 'Usage: py cheeser.py [init|load|test]'
    # Variable for HOG SGD filename
    hog_sgd_filename = 'hog_sgd_model.pkl'
    # Variable to hold fully trained model
    full_train_model = 'full_train_model.pkl'
    # Variable to hold path to indir
    path_to_indir = 'indir'
    # Variable to store the state
    state = None
    # Define the fraction of data to be trained with
    f_tr = 0.9
    # Set to True to print out examples of the data images
    print_examples = False

    # Check for number of arguments passed
    if len(argv) == 2:
        # Validate init flag
        if argv[1] == 'init':
            print("Initializing data file\n")
            # Set state to init
            state = 'init'
            # Create path to data directories
            data_path = os.path.join("data")
            # Search for training data subdirectory
            if os.path.isdir(data_path) == False:
                # Print error
                print("ERROR: Cannot find data\\ directory\n")
                # End program and return status of -1
                return -1
            # Check if there are subdirectories of data within data/
            elif len(os.listdir(data_path)) == 0:
                # Print error
                print("ERROR: data\\ subdirectory is empty\n")
                # End program and return status of -1
                return -1
            # Subdirectories of data to include
            include = {'Cheese', 'NotCheese'}
            # Check if the .pkl file exists already
            if os.path.isfile(f'{base_name}_{width}x{width}px.pkl'):
                # If it does, delete it
                os.remove(f'{base_name}_{width}x{width}px.pkl')
            print("Reading and resizing all the data images...")
            # Make new .pkl file with the data path, filename, resize width and included subdirectories
            resize_all(src=data_path, pklname=base_name, width=width, include=include)
            # If there is a fully trained model
            if os.path.isfile(full_train_model):
                # Remove the fully trained model since there are new photos to train
                os.remove(full_train_model)
                print("Deleting previous fully trained model\n")
        # If load flag is given
        elif argv[1] == 'load':
            # Enable loading sgd file instead of training new model
            state = 'load'
            # Set training fraction to be 0
            f_tr = 0
        # If testing from indir flag is given
        elif argv[1] == 'test':
            print("Testing from indir\\\n")
            # Enable testing
            state = 'test'
            # Train with all the given data
            f_tr = 1
            # Check if indir exists
            if os.path.isdir(path_to_indir) == False:
                # Print error
                print("ERROR: Cannot find indir\\")
                # Print usage
                print(usage)
                # End program and return -1
                return -1
            # Check if there are files in indir
            elif len(os.listdir(path_to_indir)) == 0:
                # Print error
                print("ERROR: No files in indir\\")
                # Print usage
                print(usage)
                # End program and return -1
                return -1
        # If more another arg is passed but it is not init flag
        else:
            # Print usgae of program
            print(usage)
            # End program with status -1
            return -1
    # If there are more than 2 argument flags
    elif len(argv) > 2:
        # Print error
        print("ERROR: Too many arguments passed")
        # Print usage
        print(usage)
        # End program and return -1
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

    # Print data informatgion
    print('number of samples: ', len(data['data']))
    print('keys: ', list(data.keys()))
    print('description: ', data['description'])
    print('image shape: ', data['data'][0].shape)
    print('labels:', np.unique(data['label']),'\n')
 
    # Make counter dict of data
    Counter(data['label'])
    
    # Use np.unique to get all unique values in the list of labels
    labels = np.unique(data['label']) 

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
            # Turn off the axis
            ax.axis('off')
            # Create variable to hold string to not interfere with fstring
            filename = 'filename'
            # Set the title
            ax.set_title(f'{data[filename][idx]}')

    # Turn the dictionary data into np arrays
    X = np.array(data['data'])
    y = np.array(data['label'])
    names = np.array(data['filename'])

    # If testing indir data
    if state == 'test':
        print("Reading images from indir\\\n")
        # Read images
        testing_data = resize_indir(path_to_indir, width=width)
        # Split indir data
        indir_data = [testing_data['data'], testing_data['label']]
        # Set testing indir to be true
        test_indir = True
    # If not testing indir data
    else:
        # Pass None
        indir_data = None
        # Set testing indir to be false
        test_indir = False
    print("Splitting training and testing data\n")
    # Split all data into training and testing based on desired ratio
    X_train, X_test, y_train, y_test, names_te = get_train_test(X=X, y=y, f_tr=f_tr, names=names, test=test_indir, indir_data=indir_data)

    # For loading SGD model
    if state == 'load':
        print(f"Reading model from {hog_sgd_filename}...\n")
        # Check to see if file exists
        if os.path.isfile(hog_sgd_filename) == True:
            # Load the file
            grid_res = joblib.load(hog_sgd_filename)
        # If not
        else:
            # Print error
            print(f"ERROR: Cannot find {hog_sgd_filename} to load from")
            # End program and return -1
            return -1
    # For testing indir data
    elif state == 'test':
        print("Searching for fully trained model...\n")
        # Check if there is a fully trained model
        if os.path.isfile(full_train_model) == True:
            print("Model found! Reading data...\n")
            grid_res = joblib.load(full_train_model)
        # If there is not a fully trained model present
        else:
            # Train the grid search to find the best descriptors
            print("Generating new fully trained grid search...\n")
            grid_res = get_grid_res(X_train, y_train)
            # Save the fully trained grid model
            joblib.dump(grid_res, full_train_model)
            print(f"New fully trained model saved as {full_train_model}\n")
    # When not testing from indir and not loading SGD model from file
    else:         
        print("Training CLF...\n")
        # Get the HOG Pipeline
        HOG_pipeline = get_HOG_pipeline()
        # Generate a Classifier with only the hog pipeline
        clf = HOG_pipeline.fit(X_train, y_train)
        # Generate a prediction with only the HOG Pipeline
        y_pred_clf = clf.predict(X_test)
        # Calculate accuracy of Pipeline transform fit
        clf_accuracy = 100*np.sum(y_pred_clf == y_test)/len(y_test)
        # Train the grid search to find the best descriptors
        grid_res = get_grid_res(X_train, y_train) 
        print("Training the grid search...\n")

    # Use the grid search results to predict the dest data
    print("Using best performing descriptors of Grid Search to predict test data...\n")
    y_pred_grid = grid_res.predict(X_test)

    # When testing from indir
    if state == 'test':
        print("HOG SGD Model Predictions:\n")
        print("Filename\tPrediction")
        # Iterate over each file
        for index in range(len(X_test)):
            # Print file and prediction
            print(f"{testing_data['filename'][index]}\t{y_pred_grid[index]}")
   
        # Get the amount of files being tested
        number_of_tests = 0
        # Iterate over each file in indir
        for file in os.listdir(path_to_indir):
            # If the file is not .DS_Store 
            if file.lower() != '.ds_store':
                # Increment the count
                number_of_tests += 1

        # If there are more than 6 testing images
        if number_of_tests > 6:
            # Number is 6
            num = 6
        # If there are less than 6 testing files
        else:
            # Number is the amount of testing files
            num = number_of_tests
        
        # Set up the matplotlib figure and axes, based on the number of labels
        fig_test, axes_test = plt.subplots(1, num)
        fig_test.suptitle(f"{num} predictions from indir\\ testing images")
        fig_test.set_size_inches(14,4)
        fig_test.tight_layout()

        # If there is only 1 axis
        if num == 1:
            # Turn it into a list
            axes_list = [axes_test]
        # If there are more than 1 axes
        else:
            # Use the pregenerated list
            axes_list = axes_test

        # Generate random indices of the predictions
        rand_indices = np.random.choice(range(number_of_tests), size=num, replace=False)
        # Iterate over each axis and index
        for ax, idx in zip(axes_list, rand_indices):
            # Display the image
            ax.imshow(testing_data['data'][idx])
            # Format the graph title
            ax.set_title(f"This is {y_pred_grid[idx]}")
            # Turn off axis tick markers
            ax.set_xticks([])
            ax.set_yticks([])
            # Set the X Label to the filename
            filename = 'filename'
            ax.set_xlabel(f"{testing_data[filename][idx]}")
        # Show the plot and wait for user to close it
        plt.show()

    # Don't compare results when reading HOG SGD model from file
    if state == 'load' or state == 'test':
        # Use the grid model predictions
        y_pred = y_pred_grid
    else:
        # Calculate accuracy of grid search
        grid_accuracy = 100*np.sum(y_pred_grid == y_test)/len(y_test)
        # Print out the accuracy of the grid search
        print(f"Grid search is {grid_accuracy}% accurate\n")
        # Print out the accuracy of the CLF Pipeline fit
        print(f"CLF is {clf_accuracy}% accurate\n")
        # Compare accuracy of Grid search vs Pipeline transform fit
        if grid_accuracy > clf_accuracy:
            # Grid search is more accurate
            y_pred = y_pred_grid
        else:
            # Pipeline is more accurate
            y_pred = y_pred_clf
        # Generate the confusion matrix
        print("Generating confusion matrix...\n")
        cmx = confusion_matrix(y_test, y_pred)
        # Plot the confusion matrices
        plot_confusion_matrix(cmx)

    # When not testing from indir
    if state != 'test':
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
        # Iterate over each testing result
        for index in range(len(y_pred)):
            # Print out the filename, true label, and the SGD model prediction
            print(f"{names_te[index]}\t{y_test[index]}\t{y_pred[index]}")
        # Print a newline
        print('')

        # Output results of the prediction
        print(f"Number of incorrect predictions: {len(incorrect_idx)} out of {len(y_pred)} examples")
        print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test),'\n')

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
         # If there are no incorrect answers
        if num != 0:
            # Set up the matplotlib figure and axes, based on the number of labels
            fig_results, axes_results = plt.subplots(1, num)
            fig_results.suptitle(f"{num} incorrect predictions from testing data")
            fig_results.set_size_inches(14,4)
            fig_results.tight_layout()

            # If there is only 1 axis
            if num == 1:
                # Turn it into a list
                axes_list = [axes_results]
            # If there are more than 1 axes
            else:
                # Use the pregenerated list
                axes_list = axes_results

            # Iterate over each axis and index
            for ax, idx in zip(axes_list, rand_indices):
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
        # If it trained a new partial sgd model
        if state != 'load':
            # Prompt user to save it
            if get_answer(f"Save HOG SGD model as {hog_sgd_filename}? (Y/N)\n", ['y','n']) == 'y':
                print("Saving HOG SGD model to .pkl file")
                # Dump the data into a .pkl file
                joblib.dump(grid_res, hog_sgd_filename)

if __name__ == '__main__':
    main()
