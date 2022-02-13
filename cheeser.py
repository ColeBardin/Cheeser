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
    testing['description'] = 'resized ({0}x{1}) testing data from ./ir in rgb'.format(int(width), int(height))
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
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True Label")

# Method to get target answer from the question
def get_answer(message, targets):
    # Prompt user
    answer = input(message)
    # Output response
    print(answer)
    # Check if response is in target answers
    for target in targets:
        # If there is a match
        if answer.lower() == target.lower():
            # Return the answer
            return answer
    # If there is no match implement recursion
    return get_answer(message, targets)

def main():
    print("\nCheesing...\n")
    # Base name of .pkl file
    base_name = 'cheese_or_not'
    # Desired image width after resize
    width = 100
    # Define usage string
    usage = 'Usage: py cheeser.py [init|load|test]'
    # State to load or train model
    load_sdg = False
    # Variable for HOG SDG filename
    hog_sdg_filename = 'hog_sgd_model.pkl'
    # Define the fraction of data to be trained with
    f_tr = 0.9
    # Status for testing indir photos
    test_indir = False
    # Variable to hold path to indir
    path_to_indir = 'indir'
    # Variable to hold fully trained model
    full_train_model = 'full_train_model.pkl'

    # Check for number of arguments passed
    if len(argv) == 2:
        # Validate init flag
        if argv[1] == 'init':
            print("Initializing data file\n")
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
        # If load flag is given
        elif argv[1] == 'load':
            # Enable loading sgd file instead of training new model
            load_sdg = True
            # Set training fraction to be 0
            f_tr = 0
        # If testing from indir flag is given
        elif argv[1] == 'test':
            print("Testing from indir\n")
            # Enable testing
            test_indir = True
            # Train with all the given data
            f_tr = 1
            # Check if indir exists
            if os.path.isdir(path_to_indir) == False:
                # Print error
                print("ERROR: Cannot find /indir")
                # Print usage
                print(usage)
                # End program and return -1
                return -1
            # Check if there are files in indir
            elif len(os.listdir(path_to_indir)) == 0:
                # Print error
                print("ERROR: No files in /indir")
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
    if test_indir == True:
        print("\nReading images from /indir")
        # Read images
        testing_data = resize_indir(path_to_indir, width=width)
        # Split indir data
        indir_data = [testing_data['data'], testing_data['label']]
    # If not testing indir data
    else:
        # Pass None
        indir_data = None
    print("\nSplitting training and testing data")
    # Split all data into training and testing based on desired ratio
    X_train, X_test, y_train, y_test, names_te = get_train_test(X=X, y=y, f_tr=f_tr, names=names, test=test_indir, indir_data=indir_data)

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
    print("\nCreating Grid Search framework\n")
    grid_search = GridSearchCV(HOG_pipeline, 
                        param_grid, 
                        cv=3,
                        n_jobs=-1,
                        scoring='accuracy',
                        verbose=1,
                        return_train_score=True)
        
    # If training a new HOG SDG file
    if load_sdg == False:
        # Check if testing indir is true
        if test_indir == True:
            # Check if there is a fully trained model
            if os.path.isfile(full_train_model) == True:
                print("Reading from fully trained model\n")
                grid_res = joblib.load(full_train_model)
            # If there is not a fully trained model present
            else:
                # Train the grid search to find the best descriptors
                print("Generating new fully trained grid search")
                # Create the grid search method
                grid_res = grid_search.fit(X_train, y_train)
                # Save the fully trained grid model
                joblib.dump(grid_res, full_train_model)
                print(f"New fully trained model saved as {full_train_model}\n")
        # Make a new model
        else:
            print("Training CLF\n")
            # Generate a Classifier with only the hog pipeline
            clf = HOG_pipeline.fit(X_train, y_train)

            # Generate a prediction with only the HOG Pipeline
            y_pred_clf = clf.predict(X_test)
            # Calculate accuracy of Pipeline transform fit
            clf_accuracy = 100*np.sum(y_pred_clf == y_test)/len(y_test)
        
            # Train the grid search to find the best descriptors
            print("Training the grid search\n")
            grid_res = grid_search.fit(X_train, y_train)
    # If loading from a file
    else:
        print(f"Reading model from {hog_sdg_filename}...\n")
        # Load the file
        grid_res = joblib.load(hog_sdg_filename)

    # Print description of best performing object
    #print(grid_res.best_estimator_)

    # Use the grid search results to predict the dest data
    print("Using best performing descriptors of Grid Search to predict test data\n")
    y_pred_grid = grid_res.predict(X_test)

    # When not testing from indir
    if test_indir == False:
        # Calculate accuracy of grid search
        grid_accuracy = 100*np.sum(y_pred_grid == y_test)/len(y_test)
        # If training a new model
        if load_sdg == False:
            # Print out the accuracy of the CLF Pipeline fit
            print(f"CLF is {clf_accuracy}% accurate\n")
        # Print out the accuracy of the grid search
        print(f"Grid search is {grid_accuracy}% accurate\n")

        # Only compare accuracies if both are performed when training new models
        if load_sdg == False:
            # Compare accuracy of Grid search vs Pipeline transform fit
            if grid_accuracy > clf_accuracy:
                # Grid search is more accurate
                y_pred = y_pred_grid
            else:
                # Pipeline is more accurate
                y_pred = y_pred_clf
        # If loading grid model
        else:
            # Use the grid model predictions
            y_pred = y_pred_grid

        # Generate the confusion matrix
        print("Generating confusion matrix\n")
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
            fig2, axes2 = plt.subplots(1, num)
            fig2.suptitle(f"{num} incorrect predictions from testing data")
            fig2.set_size_inches(14,4)
            fig2.tight_layout()

            # If there is only 1 axis
            if num == 1:
                # Turn it into a list
                axes_list = [axes2]
            # If there are more than 1 axes
            else:
                # Use the pregenerated list
                axes_list = axes2

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
    # If testing from indir
    else:
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
        fig3, axes3 = plt.subplots(1, num)
        fig3.suptitle(f"{num} predictions from /indir testing images")
        fig3.set_size_inches(14,4)
        fig3.tight_layout()

        # If there is only 1 axis
        if num == 1:
            # Turn it into a list
            axes_list = [axes3]
        # If there are more than 1 axes
        else:
            # Use the pregenerated list
            axes_list = axes3

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

    # If it trained a new sdg model
    if load_sdg == False and test_indir == False:
        # Prompt user to save it
        if get_answer(f"Save HOG SDG model as {hog_sdg_filename}? (Y/N)", ['y','n']) == 'y':
            print("Saving HOG SDG model to .pkl file")
            # Dump the data into a .pkl file
            joblib.dump(grid_res, hog_sdg_filename)

if __name__ == '__main__':
    main()
