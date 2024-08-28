import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.src.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.src.layers import Lambda, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.src.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf

from utils import INPUT_SHAPE,batch_generator

#Load data
data_dir = 'Datasets/Udacity_Self_Driving_Car/Lake'
data_df = pd.read_csv('Datasets/Udacity_Self_Driving_Car/Lake/driving_log.csv',
                   names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

#path to camera in center, left,right
x = data_df[['center', 'left', 'right']].values
#get steering of car
y = data_df['steering'].values

# plt.hist(y)
# plt.show()

#drop and get only 1000 data of steering=0
pos_zero = np.array(np.where(y==0)).reshape(-1,1)
pos_none_zero = np.array(np.where(y!=0)).reshape(-1,1)
np.random.shuffle(pos_zero)
pos_zero = pos_zero[:1000]

pos_combined = np.vstack((pos_zero, pos_none_zero))
pos_combined = list(pos_combined)

y = y[pos_combined].reshape(len(pos_combined))
x = x[pos_combined, :].reshape((len(pos_combined), 3))

# plt.hist(y)
# plt.show()

x_train, x_valid, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

#Model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(8, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(8, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(16, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='elu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(Dense(256, activation='elu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(Dense(128,activation='elu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(Dense(1, activation='tanh'))

model.summary()

nb_epoch = 15
samples_per_epoch = 1000
batch_size = 32
save_best_only = True
learning_rate = 1e-3

#this checkpoint instructs the model to save itself if validation loss is the lowest
checkpoint = ModelCheckpoint('Self_Driving_Car_Lake.keras',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=save_best_only,
                             mode='auto')

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate), metrics=['mae'])

H = model.fit(batch_generator(data_dir,x_train,y_train, batch_size, True),
          steps_per_epoch=samples_per_epoch,
          epochs=nb_epoch,
          validation_data=batch_generator(data_dir, x_valid,y_val, batch_size, False),
          validation_steps=len(x_valid) // batch_size,
          callbacks=[checkpoint],
          verbose=1)

# Plot training and validation loss
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
