import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt

df = pd.read_csv('fer2013.csv')

# Dataset info.
# 3 column (emotion(0-6), pixels, usage)
print(df.info())

# usage gives us how many data are for Training, PublicTest and PrivateTest
# Training       28709
# PublicTest      3589
# PrivateTest     3589
print(df["Usage"].value_counts())

# lets see some data, to better understand data format
print(df.head())

X_train, train_y, X_test, test_y = [], [], [], []

# now go through all the data
# if usage = 'Training' use it as a Training data
# if usage = 'PublicTest' use that data as a Test data
for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            # converting all data into float32
            # because later on we are going to normalize it,
            #   Normalize - we will take mean of those values,
            #   and subtract the mean from each data value then divide with standard deviation
            # that's why we will need values in float32

            # Also we will convert this data array(X_train, train_y, X_test, test_y) into np array,
            #   because keras accept input as np array format
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occurred at index :{index} and row:{row}")

# see if X_train... etc is working properly
print(f"X_train sample data:\n{X_train[0:2]}\n")
print(f"train_y sample data:\n{train_y[0:2]}\n")
print(f"X_test sample data:\n{X_test[0:2]}\n")
print(f"test_y sample data:\n{test_y[0:2]}\n")

num_features = 64
# num_labels use to detect emotion(0-6)
num_labels = 7
batch_size = 64
epochs = 3
width, height = 48, 48

# Now convert all those list into np array, because keras only accept input on np array format

# Didn't we do that before? Why doing it again?
#   because, previous array conversion is like np arrays of an array
#   what is that even mean?

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')


# Normalization 0 to 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

# Reshape all data
# why?
#   currently, list are containing pixels values, but after reshape it'll convert in a format that keras will take
#   basically reshaping is needed get required format keras will take as input
X_train = X_train.reshape(X_train.shape[0], width, height, 1)
X_test = X_test.reshape(X_test.shape[0], width, height, 1)
print(f"shape:{X_train.shape}")

# As we are going use 'categorical_crossentropy' to compile the model
#   so we need to change train_y and test_y accordingly

# what np_utils.to_categorical do?
#   we have values from 0 to 6 in int format,
#   np_utils.to_categorical will convert those values in matrix format, in such way that most suitable for
#   categorical_crossentropy function
train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
test_y = np_utils.to_categorical(test_y, num_classes=num_labels)


#                   designing the cnn                   #

# 1st convolution layer
model = Sequential()
# X_train.shape[1:], 0 index gives us the position of that data sample, so we don't need that
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Dropout,
#   randomly drop some data from output of this layer, so that model don't over fit
model.add(Dropout(0.5))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

# activation = 'softmax'
#   as we are doing multi class classification
model.add(Dense(num_labels, activation='softmax'))

# model.summary()
#                   Compiling the model                 #
# optimizer = Adam() as it is multiclass classification #
model.compile(loss=categorical_crossentropy, 
              optimizer=Adam(),
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("Done printing....")

#......new part for confusion matrix....
from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
 
#                      (or)

#y_pred = model.predict_classes(X_test)
#print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(BIKES)', 'class 1(CARS)', 'class 2(HORSES)']
print(classification_report(np.argmax(test_y,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(test_y,axis=1), y_pred))


#......new part for confusion matrix....

#                   Saving the model                    #
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
