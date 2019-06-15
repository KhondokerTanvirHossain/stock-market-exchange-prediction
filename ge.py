import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
import math
from sklearn.metrics import mean_squared_error
import requests
from textblob import TextBlob
#dataset_main = pd.read_csv('Google_Stock_Price_Train.csv')
#dataset = dataset_main.iloc[0:1259, 1:2].values
file = open("Stocks/ge.us.txt", "r") 
dataset = [[]]
count = 0
for line in file:
    tokens = line.split(',')
    array = [0]
    if count > 0 :
        array[0] = float(tokens[1])
        dataset.insert(count,array)
    count = count + 1
#print (count)
dataset.pop(0)
#print (dataset)    

sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset)


def train():    
    #training_set = dataset.iloc[0:4001, 2:3].values
    #training_set_scaled = sc.fit_transform(training_set)
    plt.plot(dataset, color = 'blue', label = 'Price')
    plt.title('Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    X_train = []
    y_train = []
    X_train = dataset_scaled[0:10000]
    y_train = dataset_scaled[1:10001]
    plt.plot(X_train, color = 'red', label = 'Scaled Price')
    plt.title('Scaled Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    
    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    

    model_json = regressor.to_json()
    with open("modelTempantm.json", "w") as json_file:
        json_file.write(model_json)
    regressor.save_weights("modelTempantm.h5")
    print("Saved model to disk")


def load():
    test_set = dataset[10001:10100]
    #test_set_scaled = sc.transform(test_set)
    test_set_scaled = dataset_scaled[10001:10100]
    json_file = open('modelTempantm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelTempantm.h5")
    print("Loaded model from disk")
    test_set_reshaped = np.reshape(test_set_scaled, (test_set_scaled.shape[0], test_set_scaled.shape[1], 1))
    predicted_temprature = loaded_model.predict(test_set_reshaped)
    predicted_temprature = sc.inverse_transform(predicted_temprature)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(predicted_temprature, color = 'blue', label = 'Predicted Price')
    plt.plot(test_set, color = 'red', label = 'Real Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    rmse = math.sqrt(mean_squared_error(test_set, predicted_temprature)) / 10
    print (rmse)
    
def prediction():
    #test_set = dataset_main.iloc[4001:4101, 2:3].values
    #test_set_scaled = sc.transform(test_set)
    test_set_scaled = dataset_scaled[10101:10110]
    json_file = open('modelTempantm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelTempantm.h5")
    test_set_reshaped = np.reshape(test_set_scaled, (test_set_scaled.shape[0], test_set_scaled.shape[1], 1))
    predicted_temprature = loaded_model.predict(test_set_reshaped)
    predicted_temprature = sc.inverse_transform(predicted_temprature)
    #print(predicted_temprature)
    return predicted_temprature
def senti():
    url = ('https://newsapi.org/v2/everything?q=%20ge%20stock%20market&apiKey=6e593f373865401e803d6874594f9063')
    response = requests.get(url)
    #print (response.json())
    parsed_json = response.json()
    #print(parsed_json['status'])
    array = parsed_json['articles']
    polarity = 0.0;
    count = 0;
    for i in array:
        #print(i['description'])
        blob = TextBlob(i['description'])
        count = count + 1
        polarity = polarity + blob.sentiment.polarity
    polarity = polarity / count
    #print(polarity)
    return polarity
def run():
    print('Prediction of ge Stock Price in Next 10 Days :')
    p =  prediction()
    s = senti()
    print("Date               Price")
    d = 10
    m = 1
    y = 2019
    for i in range(0,9):
        if (d == 31):
            d = 1;
            m += 1;
        if (m == 13):
            m = 1;
        print(str(d) + "-" + str(m) + "-"+ str(y)+":      "+ str(p[i][0]))
        d += 1
    print('news polarity : ' + str(s))
    if s > 0.5 :
        print('User Demand Is Very High')
    elif s > 0:
      print('User Demand Is High')
    elif s < -0.5:
        print('User Demand Is Very Low')
    elif s < 0:
        print('User Demand IS Low')        
