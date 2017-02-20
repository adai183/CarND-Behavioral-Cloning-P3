import numpy as np
import pandas as pd
import cv2
import math
import scipy.misc
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from time import time
start_time = time()

size1 = 64
size2 = 96


def process(data):
    """
    @data: pd.DataFrame
    """
    images = np.empty(shape=[data.shape[0], size1, size2, 3])
    steering_angles = np.empty(shape=[data.shape[0]])

    i = 0
    for index, row in data.iterrows():

        # Preprocess image
        fn = row.name
        img = cv2.imread('Data/IMG/' + fn)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Full color
        img = img[:, :, :].squeeze()
        # crop the image
        img = img[40:160, 10:310, :]
        # resize the image maintaining aspect ratio
        img = cv2.resize(img, (size2, size1))
        # # normalize the image
        img = (img / 127.5) - 1

        # randomly choose to flip the image, and invert the steering angle.
        if np.random.uniform() > 0.5 and row['steering'] != 0.0:
            images[i, :, :, :] = cv2.flip(images[i, :, :, :], 1)
            row['steering'] = - row['steering']

        steering_angles[i] = row['steering']

        images[i, :, :, :] = img
        i += 1

    return images, steering_angles


def generator(iterable, batch_size=512):
    """
    @iterable: pd.DataFrame
    """

    img_num = iterable.shape[0]

    while True:
        # shuffle Data before creating batches
        iterable = iterable.sample(frac=1)

        for ndx in range(0, img_num, batch_size):
            batch = iterable.iloc[ndx:min(ndx + batch_size, img_num)]

            images, steering_angles = process(batch)

            yield (images, steering_angles)


model = Sequential()

# 0. Model chooses how to convert Color channels
model.add(Convolution2D(1, 1, 1, border_mode='same',
                        init='glorot_uniform', input_shape=(size1, size2, 3)))


# 1. Convolutional, kernel(3,3) - Output 24
model.add(Convolution2D(24, 3, 3, border_mode='valid',
                        init='glorot_uniform', W_regularizer=l2(0.01)))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

# 2. Convolutional, kernel(3,3) - Output 36
model.add(Convolution2D(36, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

# 3. Convolutional, kernel(3,3) - Output 48
model.add(Convolution2D(48, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

# # 4. Convolutional, kernel(3,3) - Output 64
# model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_uniform'))
# model.add(Activation('elu'))

# 4. Flatten and dropout
model.add(Flatten())
model.add(Dropout(.5))

# # 6. Dense 1064
# model.add(Dense(1064, init='uniform'))
# model.add(BatchNormalization())
# model.add(Activation('elu'))

# 5. Dense 100
model.add(Dense(100, init='uniform'))
# model.add(BatchNormalization())
model.add(Activation('elu'))


# 6. Dense 50
model.add(Dense(100, init='uniform'))
# model.add(BatchNormalization())
model.add(Activation('elu'))

# 7. Dense 10
model.add(Dense(10, init='uniform'))
# model.add(BatchNormalization())
model.add(Activation('elu'))

# 8. Output
model.add(Dense(1))


LEARNING_RATE = 0.001
EPOCHS_NUM = 100
BATCH_SIZE = 512

# minimum change in validation loss after each epoch to qualify as an
# improvement
DELTA = 0.001


model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
print ('Model compiled.')


# Load validation data
measurements_valid = pd.DataFrame.from_csv('Data/valid_data.csv')
X_valid, y_valid = process(measurements_valid)

# Load Dataframe for training data
measurements_train = pd.DataFrame.from_csv('Data/train_data.csv')

batch_num = math.ceil(measurements_train.shape[0] / BATCH_SIZE)

# model.fit_generator(generator(measurements_train, batch_size=BATCH_SIZE),
#                     samples_per_epoch=measurements_train.shape[0],
#                     nb_epoch=EPOCHS_NUM,
#                     verbose=1,
#                     callbacks=[EarlyStopping(monitor='val_loss',
#                                              min_delta=DELTA,
#                                              patience=2,
#                                              verbose=2,
#                                              mode='min'),
#                                ModelCheckpoint('best_model.h5',
#                                                monitor='val_loss',
#                                                verbose=1,
#                                                save_best_only=True,
#                                                mode='min',
#                                                period=1),
#                                CSVLogger('train_stats.csv')
#                                ],
#                     validation_data=(X_valid, y_valid),
#                     nb_val_samples=y_valid.shape[0])



end_time = time()
time_taken = end_time - start_time  # time_taken is in seconds

hours, rest = divmod(time_taken, 3600)
minutes, seconds = divmod(rest, 60)

print ("Time: ", hours, "h, ", minutes, "min, ", seconds, "s ")
