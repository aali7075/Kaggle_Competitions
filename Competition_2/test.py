"""
Aaron Li
Ean Kramer
"""


import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from feature_selector import FeatureSelector

#create DF and split into train and test
stockDF = pd.read_csv("stock_XY_train.csv")
fs = FeatureSelector()
# stockDF = stockDF.dropna(axis='columns', thresh=stockDF.shape[0]*.8)
# stockDF = stockDF.dropna()
# stockDF = stockDF.drop(stockDF.columns[[0,1]], axis=1)
# stockDF_y = stockDF["Buy"]
# stockDF = stockDF.drop(columns=["Yr", "Buy", "Sector"])
# scaler = preprocessing.MinMaxScaler()
# stockDF = scaler.fit_transform(stockDF)
# train_X, val_X, train_y, val_y = train_test_split(stockDF, stockDF_y, train_size=.8, shuffle=True)

#testing shit
print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)
# print(stockDF.dtypes)

#create initial model
model = keras.Sequential()

#add layers to model
model.add(Dense(64, input_dim= stockDF.shape[1], activation="relu"))
model.add(Dropout(.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(.2))
model.add(Dense(1, activation="sigmoid"))

#compile model
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
model.fit(train_X,train_y, validation_data=(val_X, val_y), epochs=10)


