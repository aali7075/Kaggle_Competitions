import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


#load and split to train/val
data = np.load("train_and_test.npz")
X_test = data.f.X_test #collapse to greyscale
# X_train, X_val, y_train, y_val = train_test_split(data.f.X_train, data.f.y_train, train_size=.8, stratify=data.f.y_train, shuffle=True)
X_train = data.f.X_train
y_train = data.f.y_train

#one hot encoding
y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(.4))
model.add(layers.Dense(43, activation='softmax'))

model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy', "categorical_crossentropy"])

# model.fit(X_train,y_train,epochs=12,verbose=1, batch_size=256, validation_data=(X_val,y_val))
model.fit(X_train,y_train,epochs=15,verbose=1, batch_size=256)
predictions = model.predict(X_test)
id = np.arange(0,predictions.shape[0])
my_ans = {"id": id}
for i in range(predictions.shape[1]):
    key = "oh_" + str(i)
    my_ans[key] = predictions[:,i]

# for key, val in my_ans.items():
#     if(key == "id"):
#         continue
#     a = np.zeros(len(val))
#     ind = np.argmax(val)
#     a[ind] = 1
#     my_ans[key] = a

ans_df = pd.DataFrame(my_ans)
ans_df.to_csv("samples.csv", index=False)
