import math
import tensorflow as tf
import numpy as np
import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers,models,Sequential
from tensorflow.keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source = 'yahoo', start = '2012-01-01', end = '2019-12-17')

plt.figure(figsize=(16,8))
plt.title('Stock(close)')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.show()

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i <= 65:
        print(x_train)
        print(y_train)
        print()
        
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Building LSTM

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1],1)))    
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

test_data = scaled_data[training_data_len-60:,:]

x_test = []
y_test = dataset[training_data_len:,:]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

rmse = np.sqrt(np.mean(pred-y_test)**2)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = pred
plt.figure(figsize=(16,8))
plt.title('Model')

plt.xlabel('Date')
plt.ylabel('Close Price USD')

plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

plt.legend(['Train','Actual value','Predictions'],loc = 'lower right')
plt.show()