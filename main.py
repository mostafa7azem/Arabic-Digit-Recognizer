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
from sklearn.model_selection import train_test_split


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
# Load the model
model = load_model('model.h5')

predictions = model.predict_classes(x_test)
loss, accuracy = model.evaluate(x_test, y_test_categorical)
print(predictions)
print(y_test)
print("Loss: " + str(loss))
print("Accuracy (%): " + str(accuracy*100))

for i in range(len(x_test)):
    cv2.imshow("Actual: " + str(y_test[i]) + " Predicted: " + str(predictions[i]), x_test[i])
    cv2.waitKey()
    cv2.destroyAllWindows()
