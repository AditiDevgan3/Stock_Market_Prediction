import math
import tkinter
import streamlit as st
from datetime import date
import datetime as dt
import tensorflow as tf
import numpy as np
import pandas_datareader as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers,models,Sequential
from tensorflow.keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
import altair as alt
plt.style.use('fivethirtyeight')

start = dt.datetime(2012,1,1,0,0)
end = dt.date.today()

st.title('Stock Market Prediction App')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

df = web.DataReader(selected_stock,'yahoo', start, end)


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

#st.write(plt.figure(figsize=(16,8)))

#st.altair_chart(alt.Chart(df['Close']).mark_line().encode(x='Date',y='Close Price USD'))
st.subheader('Stock(close)')
st.line_chart(df['Close'])

#st.write(plt.show())

data = df.filter(['Close'])
dataset = data.values
st.write(data)
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

#Building_LSTM

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
st.subheader('Model')

plt.xlabel('Date')
plt.ylabel('Close Price USD')

plt.plot(train['Close'])
st.line_chart(valid[['Close','Predictions']])

plt.legend(['Train','Actual value','Predictions'],loc = 'lower right')
plt.show()
