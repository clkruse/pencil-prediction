import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

filename = 'test.json'
with open(filename) as f:
  data = json.load(f)

x = []
y = []

for stroke_num in range(1,501):
    stroke_x = [entry['x'] for entry in data if entry['strokeNum'] == stroke_num and entry['lineWidth'] != 0]

    stroke_y = [-1 * entry['y'] for entry in data if entry['strokeNum'] == stroke_num and entry['lineWidth'] != 0]

    norm_x = [value - stroke_x[0] for value in stroke_x]
    x.append(norm_x)

    norm_y = [value - stroke_y[0] for value in stroke_y]
    y.append(norm_y)

from keras.preprocessing.sequence import pad_sequences

x_pad = pad_sequences(x, padding='post', maxlen=35)
y_pad = pad_sequences(y, padding='post', maxlen=35)
x_pad[0]
y_pad[0]
x_input = np.array([x_pad, y_pad])
x_input.shape
x_input = np.moveaxis(x_input, 0, -1)

x = []
y = []
for i in range(len(x_input)):
    x = x_input[:,:30,:]
    y = x_input[:,5:,:]

for i in range(5):
    plt.plot(x[i,:,0], x[i,:,1])
    plt.plot(y[i,:,0], y[i,:,1])
    plt.show()

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(TimeDistributed(Dense(2, activation='linear')))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x, y, epochs=500)

preds = model.predict(x[0:100])
preds.shape
for i in range(5):
    plt.plot(x[i,:,0], x[i,:,1])
    plt.plot(preds[i,:,0], preds[i,:,1], color='r')
    plt.show()


plt.figure(figsize=[10,8])
for stroke_num in range(501):
    x_val = [entry['x'] for entry in data if entry['strokeNum'] == stroke_num and entry['lineWidth'] != 0]

    norm_x = [(entry['x'] - min(entry))/(max(entry) - min(entry)) for entry in data if entry['strokeNum'] == stroke_num and entry['lineWidth'] != 0]

    seq_lengths.append(len(x_val))

    y_val = [-1 * entry['y'] for entry in data if entry['strokeNum'] == stroke_num and entry['lineWidth'] != 0]

    pressure = [(entry['pressure'] * 10) ** 2 for entry in data if entry['strokeNum'] == stroke_num and entry['lineWidth'] != 0]
    plt.plot(x_val, y_val, color='k')
    #plt.scatter(x_val, y_val, s = pressure, c=pressure)
plt.show()
