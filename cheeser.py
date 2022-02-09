import numpy as np
from sklearn import datasets, svm

#Split dataset into training vs testing data. usually split 50/50 randomly is sufficient
def split_train_test (X, y, f_tr):
    # get number of instances
    n = len(X)
    # set number of images for training , testing
    n_tr = int(f_tr * n)
    n_te = n - n_tr
    # pick indices for training
    i_tr = np.random.choice(n, n_tr, replace=False)
    # split X_lst into training and testing
    X_tr = [X[i] for i in range(n) if i in i_tr]
    X_te = [X[i] for i in range(n) if i not in i_tr]
    # split y_lst into training and testing
    y_tr = [y[i] for i in range(n) if i in i_tr]
    y_te = [y[i] for i in range(n) if i not in i_tr]
    # return training and testing
    return X_tr , X_te , y_tr , y_te


def main():
    #dataset = datasets.load_digits()
    #X_im=dataset.images
    #y=dataset.target
    num = 1000
    X_im = [[i] for i in range(num)]
    y = []
    for i in range(num):
        if i%2 == 1:
            y.append(1)
        else:
            y.append(0)
    #print(X_im)
    #print(y)

    X_tr , X_te , y_tr , y_te = split_train_test(X_im, y, 0.8)
    # create a support vector classifier (SVC)
    cls = svm.SVC()
    # train the classifier with training data
    cls.fit(X_tr , y_tr)
    # use classifier to predict on test X_te
    z_te = cls.predict(X_te)
    print()

    z_te_incorrect = []
    # print table of mis-classified test values
    print("Misclassififed Test Values")
    print("i\txt\ty\tz")
    for i in range(len(y_te)):
        if y_te[i] != z_te[i]:
            print("{}\t{}\t{}\t{}".format(i,X_te[i][0],y_te[i],z_te[i]))
            z_te_incorrect.append([i,y_te,z_te])
    print(f"Number of incorrect guesses: {len(z_te_incorrect)}")

if __name__ == '__main__':
    main()