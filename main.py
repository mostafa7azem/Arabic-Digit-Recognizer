import os
import cv2
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
import h5py
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.models import Model
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


data_path = os.path.join("Data", "BlackMasked")
imgs = os.listdir(data_path)

images = []
digits = []
for img in imgs:
    image = cv2.imread(os.path.join(data_path, img))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    images.append(image)
    digits.append(img.split('_')[0])

np_images = np.array(images)
np_digits = np.array(digits)

np_images = np_images.reshape(-1, 64, 48, 1)

x_train, x_test, y_train, y_test = train_test_split(np_images, np_digits, test_size=0.1, random_state=0)
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)
knn = KNeighborsClassifier(n_neighbors=10)
dataset_size = len(x_train)
TwoDim_dataset = x_train.reshape(dataset_size,-2)
print(TwoDim_dataset,y_train)
knn.fit(TwoDim_dataset, y_train)
dataset_size = len(x_test)
TwoDim_dataset2= x_test.reshape(dataset_size,-2)
print(knn.score(TwoDim_dataset2, y_test))
print("test dataset : values =")
print(y_test)
print("predicted  : values =")
print(knn.predict(TwoDim_dataset2))
for i in range(len(x_test)):
    cv2.imshow("Actual: " + str(y_test[i]) + " Predicted: " + str(knn.predict(TwoDim_dataset2)[i]), x_test[i])
    cv2.waitKey()
    cv2.destroyAllWindows()

'''
model = Sequential()
model.add(Conv2D(64, 7, activation='relu', input_shape=(64, 48, 1)))
model.add(MaxPooling2D(2))

model.add(Conv2D(64, 7, activation='relu'))
model.add(MaxPooling2D(2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

early_stop = EarlyStopping(patience=3)
checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
ModelCheckpoint
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(x_train, y_train_categorical, epochs=7, validation_split=0.1, callbacks=[checkpoint, early_stop])
model.save('model.h5')

model = load_model('model.h5')
[node.op.name for node in model.outputs]
print(model.summary())
predictions = model.predict_classes(x_test)
loss, accuracy = model.evaluate(x_test, y_test_categorical)
print(predictions)
print(y_test)
print("Loss: " + str(loss))
print("Accuracy (%): " + str(accuracy*100))
'''
'''
for i in range(len(x_test)):
    cv2.imshow("Actual: " + str(y_test[i]) + " Predicted: " + str(predictions[i]), x_test[i])
    cv2.waitKey()
    cv2.destroyAllWindows()
'''