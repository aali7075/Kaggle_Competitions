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
stockTest = pd.read_csv("stock_X_test.csv")
stockDF = stockDF.dropna(axis='columns', thresh=stockDF.shape[0]*.75)
# stockDF = stockDF.dropna()
stockDF = stockDF.drop(stockDF.columns[[0,1]], axis=1)
stockTest_names = stockTest["labels"]
stockTest = stockTest.drop(stockTest.columns[[0,1,2]], axis=1)
stockDF_y = stockDF["Buy"]
stockDF = stockDF.drop(columns=["Yr", "Buy", "Sector"])
stockTest = stockTest.drop(columns=["Yr", "Sector"])


"""TRYING OUT A TREE MODEL AND FINDING MOST IMPORTANT FEATURES"""

regr = AdaBoostRegressor(random_state=0, learning_rate = .5,
base_estimator=DecisionTreeRegressor(max_depth=16, max_features="sqrt"),
n_estimators=13, loss = "linear")
stockDF = stockDF.fillna(stockDF.mean())
regr.fit(stockDF,stockDF_y);

feature_names= stockDF.columns
feature_dict = { feature_names[i]: regr.feature_importances_[i] for i in range(len(feature_names))}
columns_tuple=[(v,k) for k,v in feature_dict.items()]
columns_tuple.sort(key=lambda tup: tup[0], reverse=True)
most_important_features=[name for val,name in columns_tuple]
# print(len(most_important_features))

stockDF=stockDF[most_important_features[0:10]]
stockTest=stockTest[most_important_features[0:10]]
stockDF=(stockDF-stockDF.mean())/stockDF.std() #normailize data
stockTest=(stockTest-stockTest.mean())/stockTest.std() #normailize data
stockTest = stockTest.fillna(stockTest.mean())
train_X, val_X, train_y, val_y = train_test_split(stockDF, stockDF_y, train_size=.8, shuffle=True)

#testing!
print(train_X.shape, val_X.shape, train_y.shape, val_y.shape)
print(stockDF.dtypes)

#create initial model
model = keras.Sequential()

#add layers to model
model.add(Dense(35, input_dim= stockDF.shape[1], activation="relu"))
model.add(Dropout(.15))
model.add(Dense(16, activation="relu"))
model.add(Dropout(.15))
model.add(Dense(32, activation="relu"))
model.add(Dropout(.15))
model.add(Dense(1, activation="sigmoid"))

#compile model
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
model.fit(train_X,train_y, validation_data=(val_X, val_y), epochs=20)

#test and output model
pred = model.predict(stockTest)
print(stockTest_names.shape)
pred_labels = []
print(pred)
for arr in pred:
    if np.isnan(arr[0]) ==  True:
        pred_labels.append(1)
    elif arr[0] < .5:
        pred_labels.append(0)
    else:
        pred_labels.append(1)
print(pred_labels)
pred_labels = np.asarray(pred_labels)
output_dict = {"Unnamed: 0" : stockTest_names, "Buy" : pred_labels}
output = pd.DataFrame(data = output_dict)
output.to_csv("submission.csv", index=False)