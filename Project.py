import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras import datasets, layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

(x_train,y_train), (x_test,y_test) = datasets.cifar10.load_data()
x_train.shape

classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
y_train_index = to_categorical(y_train)
y_test_index = to_categorical(y_test)
print(y_train_index)

x_train = x_train / 255
x_test = x_test / 255
x_train[0]

model = Sequential()
model.add( Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)) )
model.add(MaxPooling2D(pool_size = (2,2)))
model.add( Conv2D(32,(5,5),activation='relu') )
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(800,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(400,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

TrainedModel = model.fit(x_train,y_train_index,
                 batch_size = 256,
                 epochs = 10)

model.evaluate(x_test,y_test_index)[1]

plt.plot(TrainedModel.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show

plt.plot(TrainedModel.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show

cvImg = plt.imread("C:/Users/OMAIR/Downloads/index.jpg")
new_size = cv2.resize(cvImg,(32,32))
print(cvImg.shape)
plt.imshow(cvImg)

print(new_size.shape)
plt.imshow(new_size)

predictions = model.predict(np.array([new_size]))
predictions

predict_list=[0,1,2,3,4,5,6,7,8,9]
X = predictions

for i in range(10):
    for j in range(10):
        if X[0][predict_list[i]] > X[0][predict_list[j]]:
            swap = predict_list[i]
            predict_list[i] = predict_list[j]
            predict_list[j] = swap
print(predict_list)

for i in range(10):
    print('It is',classification[predict_list[i]],':',round(predictions[0][predict_list[i]]*100,2),'%')
