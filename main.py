import os

from skimage.io import imread
from skimage.transform import resize

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

#To save models
import pickle

## prepare data
#print(__file__) #The full path to the current file
# Gets the folder that contains that file (ends with .something) and prevents returning an empty string
dirname = os.path.dirname(os.path.abspath(__file__))
#print(dirname)
input_dir = os.path.join(dirname, "data") #os.path.join(dirname, 'relative/path/to/file/you/want')
#print(input_dir)
categories = ['empty', 'not_empty']

data = []
labels = []
#Enumerate number and value. Ex: 0 - empty, 1 - not_empty.
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        #print(img_path) #See if the path is outputting in the correct format
        # Loads the image from the path        
        img = imread(img_path)
        #Resize the image, so it is easier to analyze and has a smaller file size
        img = resize(img, (15, 15))
        #Stores in the images loaded (an array called "data")
        #Reduces the three channel info for 15 X 15 pixels into one large unidimensional array.
        data.append(img.flatten())
        #Remember which category the image was.
        labels.append(category_idx)

#Converts the loaded data into numpy array, because it is more efficient and
data = np.asarray(data)
labels = np.asarray(labels)

## train / test split
#Gets a large dataset and divides it into some groups of data to used as the data to be used to train or to test if results are right
#test_size - percentage of how much of the random data is going to be used to test, the rest is for training. (0.2 is 20%)
#shuffle - makes the data random, to prevent bias
#stratify - stratifying according to the data makes the data always proportional according to the labels. Even if it is random
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

## train classifier
classifier = SVC()

# For SVC classifier (C-Support Vector Classification) there is just two parameters.
parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]

#Exaustive search over specified parameters values for an estimator. Implements fit and score method
grid_search = GridSearchCV(classifier, parameters)
#Runs fit with x array, the y array and all sets of parameters.
grid_search.fit(x_train, y_train)

## test performance
#Gets the best of all the image classifiers that were found
best_estimator = grid_search.best_estimator_
#The best estimator were CSV with C=10 and gamma=0.01.
#print(best_estimator)

#Performs classification on samples in X. Returns y_pred = "Class labels" for samples in X
y_prediction = best_estimator.predict(x_test)

#Accuracy classification score.
score = accuracy_score(y_prediction, y_test)

print(f"{score * 100}% of samples were correctly classified.") #Score gives a number between 0 to 1 that is a percentage. (0.1, or 0.3 for example)

#Saves the model in the current path with mode.p extension
#https://stackoverflow.com/questions/2665866/what-does-wb-mean-in-this-code-using-python
#The Python programming language also utilizes this file extension to store Python module files which are not yet in the byte stream format.
#https://www.online-convert.com/file-format/p#:~:text=P%20files%20contain%20picture%20files,runtime%20files%20within%20the%20application.
pickle.dump(best_estimator, open("./model.p", "wb"))
