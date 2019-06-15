# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import requests
url = ('https://newsapi.org/v2/everything?q=%20google%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
print (response.json())

import requests
url = ('https://newsapi.org/v2/everything?q=%20Apple%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
print (response.json())

import requests
url = ('https://newsapi.org/v2/everything?q=%20Microsoft%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
print (response.json())

import requests
url = ('https://newsapi.org/v2/everything?q=%20IBM%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
response = requests.get(url)
print (response.json())

import json

dt = {}
dt = response.json()
st=""
for key,val in dt.items():
    st = st + key


dataset = pd.read_csv('newss.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 0:1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
y[:, 0] = labelencoder_X_1.fit_transform(y[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 0 : 11]
y = y[:, 0]
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
def ret():
    return y_pred[0][0]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)