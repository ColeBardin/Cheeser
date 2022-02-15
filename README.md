# Cheeser

Is this cheese?

Image recognition software that trains to detect images of cheese using SciKit-Learn models

Trains with images of cheese and not cheese then makes predictions on a fraction of the sample photos

## cheeser.py

Usage: `py cheeser.py [init|load|test]`

Train software with photos of cheese and photos of not cheese

Use the directores `data\Cheese\` and `data\NotCheese\` to train the AI

Optional `init` commandline argument will reload all the images into the `.pkl` file before training. This only needs to be run if there new photos have been added. Without the `init` flag, cheeser.py will attempt to read from a `.pkl` file composed of the base name and the dimensions

Optional `load` flag will not train a new HOG SGD model, but instead read one in from  `hog_sgd_model.pkl`

Optional `test` flag will read in testing data from `indir\`. When called, the program will try to find a fully trained model under the name `full_train_model.pkl`. If does not exist, it will train one with all the images in `data\` and create a `.pkl` file for it

Specifying no flags will split the data from `data\` to 90% for training and 10% for testing. Then it will create a CLF and Grid Search model, compare the two and display the results. Lastly, the program will give you the option to save the trained model as `hog_sgd_model.pkl`

## Dependencies

### cheeser.py:

[Matplotlib](https://pypi.org/project/matplotlib/)

[Joblib](https://pypi.org/project/joblib/)

[Numpy](https://pypi.org/project/numpy/)

[Scikit-Learn](https://pypi.org/project/scikit-learn/)

[Scikit-Image](https://pypi.org/project/scikit-image/)

### transformer_classes.py:

[Numpy](https://pypi.org/project/numpy/)

[Scikit-Learn](https://pypi.org/project/scikit-learn/)

[Scikit-Image](https://pypi.org/project/scikit-image/)
