# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import pickle
import keras

class PlotLosses(keras.callbacks.Callback):
  
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

tmp = []
data = []


with open('path_to_your_dataset.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')
    row_count = 0

    for row in csv_reader:
        row_count += 1

data_index = 0

train = np.zeros((row_count-30, 1))
test = np.zeros((30, 1))
data = np.zeros((row_count, 1))
reverse_data = np.zeros((row_count, 1))

with open('path_to_your_dataset.csv') as csv_file:

    counter = 0
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        data[counter, 0] = (row[5])
        counter += 1

# scaler = MinMaxScaler(feature_range=(0, 1))
# data = scaler.fit_transform(data)

max_y = max(data)
min_y = min(data)

range_n = max_y - min_y


for i in range(0, len(data)):
    reverse_data[row_count - i - 1] = ((data[i] - min_y) / (range_n))


for i in range(0, len(reverse_data)-30):
    train[i, 0] = reverse_data[data_index]
    data_index += 1

index = 0


for i in range(len(reverse_data)-30, len(reverse_data)):

    test[index, 0] = reverse_data[data_index]
    index+=1
    data_index += 1


def create_dataset(dataset, look_back):
    
    dataX, dataY = [], []

    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)


look_back = 3

train_x, train_y = create_dataset(train, look_back)

test_x, test_y = create_dataset(test, look_back)

data_x, data_y = create_dataset(reverse_data, look_back)


# -->if you wanna use a Dense layer at first, comment three following lines:

# train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
# test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
# data_x = np.reshape(data_x, (data_x.shape[0], 1, data_x.shape[1]))

regressor = Sequential()

#--> 1) Dense


# regressor.add(Dense(units = 32, activation = 'relu'))

# regressor.add(Dense(units = 32, activation = 'relu'))


#--> 2) LSTM

regressor.add(LSTM(50,  activation='relu', return_sequences=True, input_shape=(1, look_back)))
regressor.add(Dropout(0.2))

# regressor.add(LSTM(50,  activation='relu', return_sequences=True))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(50,  activation='relu', return_sequences=True))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(50,  activation='relu', return_sequences=True))
# regressor.add(Dropout(0.2))


# --> 'return_sequences' param must be true for all the LSTM layers except the last one!

regressor.add(LSTM(50,  activation='relu'))
regressor.add(Dropout(0.2))



#--> 3) GRU

# regressor.add(GRU(50, activation='tanh', return_sequences=True, input_shape=(1, look_back)))
# regressor.add(Dropout(0.2))

# regressor.add(GRU(50, activation='tanh', return_sequences=True))
# regressor.add(Dropout(0.2))

# regressor.add(GRU(50, activation='tanh', return_sequences=True))
# regressor.add(Dropout(0.2))

# regressor.add(GRU(50, activation='tanh', return_sequences=True))
# regressor.add(Dropout(0.2))

# --> 'return_sequences' param must be true for all the GRU layers except the last one!

# regressor.add(GRU(50, activation='tanh', return_sequences=False))
# regressor.add(Dropout(0.2))



# final layer of all models

regressor.add(Dense(units = 1, activation = 'relu'))


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])


# --> set the path of your model

filepath="/tmp2/model_LSTM_2.Saipa"


tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp2/LSTM_2', histogram_freq=1)


checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')  


hist = regressor.fit(train_x, train_y, epochs = 100, batch_size = 500,
                      validation_data=(test_x, test_y),
          callbacks=[plot_losses, checkpoint, tbCallBack],
          verbose=0)


regressor.summary()


print(hist.history.keys())


predicted_stock_price = regressor.predict(test_x)


train_predict_price = regressor.predict(train_x)

from sklearn.metrics import mean_squared_error, mean_absolute_error

predicted_stock_price = regressor.predict(test_x)
train_predict_price = regressor.predict(train_x)

print('test info: ')
print('MSE is: ')
print(mean_squared_error(test_y, predicted_stock_price))
print('MAE is: ')
print(mean_absolute_error(test_y, predicted_stock_price))

print('train info: ')
print('MSE is: ')
print(mean_squared_error(train_y, train_predict_price))
print('MAE is: ')
print(mean_absolute_error(train_y, train_predict_price))

plt.plot(test_y, label="test data")
plt.plot(predicted_stock_price, label="prediction of test data")
plt.xlabel('day')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()

# load model and predict the stock price for following 30 days:
# don't forget to set the path to your best model

filepath="bestModel.h"

from keras.models import load_model

model = load_model(filepath)

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

# finally predict the stock price for following 30 days with your best model:

print(data_x[-1:,])

def predict_30days_after(token):
  
  preds = []

  def rebuild(data, pred):

    data[0][0] = data[0][1]
    data[0][1] = data[0][2]
    data[0][2] = pred

    return data

  for i in range(0, 30):

    pred = model.predict(token)[0][0]
    preds.append(pred)
    token = rebuild(token, pred)

  return ((preds * range_n) + min_y)

print(predict_30days_after(data_x[-1:,]))
