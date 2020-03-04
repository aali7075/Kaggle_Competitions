"""
Aaron Li
Ean Kramer
"""


import numpy as np
import pandas as pd
import keras
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD

#create DF and split into train and test
stockDF = pd.read_csv("stock_XY_train.csv")
stockDF = stockDF.dropna(axis='columns', thresh=stockDF.shape[0]*.9)
stockDF = stockDF.dropna()
stockDF = stockDF.drop(stockDF.columns[[0,1]], axis=1)
stockDF_y = stockDF["Buy"]
stockDF = stockDF.drop(columns=["Yr", "Buy", "Sector"])



"""TRYING OUT A TREE MODEL AND FINDING MOST IMPORTANT FEATURES"""

regr = AdaBoostRegressor(random_state=0, learning_rate = .5,
base_estimator=DecisionTreeRegressor(max_depth=16, max_features="sqrt"),
n_estimators=13, loss = "linear")
regr.fit(stockDF,stockDF_y);

feature_names= stockDF.columns
feature_dict = { feature_names[i]: regr.feature_importances_[i] for i in range(len(feature_names))}
columns_tuple=[(v,k) for k,v in feature_dict.items()]
columns_tuple.sort(key=lambda tup: tup[0], reverse=True)
most_important_features=[name for val,name in columns_tuple]
# print(len(most_important_features))

stockDF=stockDF[most_important_features[0:35]]
stockDF=(stockDF-stockDF.mean())/stockDF.std() #normailize data
train_X, val_X, train_y, val_y = train_test_split(stockDF, stockDF_y, train_size=.8, shuffle=True)

#testing!
print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)
print(stockDF.dtypes)

#create initial model
model = keras.Sequential()

#add layers to model
model.add(Dense(50, input_dim= stockDF.shape[1], activation="relu"))
model.add(Dense(30, activation="relu"))
model.add(Dropout(.2))
model.add(Dense(10, activation="relu"))
model.add(Dense(40, activation="relu"))
model.add(Dropout(.05))
model.add(Dense(5, activation="relu"))


model.add(Dense(1, activation="sigmoid"))

#compile model
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
model.fit(train_X,train_y, validation_data=(val_X, val_y), epochs=80)
