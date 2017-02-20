import numpy as np
import pandas as pd
import cv2
import math
from termcolor import colored
import csv 

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback, EarlyStopping

from time import time
start_time = time()


# def get_memory_stats():
#     stats = dict()
#     for var, obj in globals().items():
#         stats[var] = getsizeof(obj)

#     stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
#     stats = pd.DataFrame(stats)

#     return stats


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
    @data: pd.DataFrame
    """

    img_num = iterable.shape[0]

    # shuffle Data before creating batches
    iterable = iterable.sample(frac=1)

    for ndx in range(0, img_num, batch_size):
        batch = iterable.iloc[ndx:min(ndx + batch_size, img_num)]

        #images, steering_angles = process(batch)
        yield batch


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
# model.add(Activation('elu'))

# 5. Dense 100
model.add(Dense(100, init='uniform'))
model.add(Activation('elu'))


# 6. Dense 50
model.add(Dense(100, init='uniform'))
model.add(Activation('elu'))

# 7. Dense 10
model.add(Dense(10, init='uniform'))
model.add(Activation('elu'))

# 8. Output
model.add(Dense(1))


LEARNING_RATE = 0.001
EPOCHS_NUM = 100
BATCH_SIZE = 512

# minimum change in validation loss after each epoch to qualify as an
# improvement
DELTA = 0.0


model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
print ('Model compiled.')


# Keras callbacks for logging and early termination
class History(Callback):
    """
    Takes care of logging
    """

    def on_train_begin(self, logs={}):
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        loss_temp = logs.get('loss')
        val_loss_temp = logs.get('val_loss')

        # save stats to csv
        with open("stats.csv", "a") as csv_file:
            data = [epoch_i, loss_temp, val_loss_temp]
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(data)

        # log stats
        print ('Epoch {}  :  Batch {}/{}'.format(
            epoch_i + 1, batch_i, batch_num))
        print ('Loss:{}  Validation Loss:{}'.format(
            loss_temp, val_loss_temp))

        save validation loss after every epoch for early termination
        if batch_i == batch_num - 1:
            self.val_losses.append(val_loss_temp)

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        if len(self.val_losses) >= 2:
            gain = self.val_losses[-2] - self.val_losses[-1]
            print ('Validation Gain: {}'.format(gain))
            if gain < DELTA:
                print (colored('Early Termination !!!', 'red'))
                quit()

# with open('model.json', 'w') as outfile:
#     outfile.write(model.to_json())

# To open previous model weights
# model.load_weights("model.h5")


# Load validation data
measurements_valid = pd.DataFrame.from_csv('Data/valid_data.csv')
X_valid, y_valid = process(measurements_valid)

# Load Dataframe for training data
measurements_train = pd.DataFrame.from_csv('Data/train_data.csv')

batch_num = math.ceil(measurements_train.shape[0] / BATCH_SIZE)
# Keep track of validation losses to implement early termination
val_losses = []
gain = 0

for epoch_i in range(EPOCHS_NUM):

    print (colored('Start epoch: {}/{}'.format(epoch_i + 1, EPOCHS_NUM),
                   'cyan'))

    batch_i = 0
    for batch in generator(measurements_train, batch_size=BATCH_SIZE):

        X_train, y_train = process(batch)

        model.fit(X_train, y_train,
                  batch_size=y_train.shape[0],
                  nb_epoch=1,
                  validation_data=(X_valid, y_valid),
                  verbose=0,
                  callbacks=[History()])

        batch_i += 1

    # handle early termination
    if len(val_losses) >= 2:
        gain = val_losses[-2] - val_losses[-1]
        print ('Validation Gain: {}'.format(gain))
        if gain < DELTA:
            print (colored('Early Termination !!!', 'red'))
            quit()

    # save model after every epoch with improvement
    model.save('model.h5')
    print ('Model saved')


# #pred  = model.predict(X_test, batch_size=128)
# #print("Test error: ", np.mean(np.square(pred-Y_test)))

end_time = time()
time_taken = end_time - start_time  # time_taken is in seconds

hours, rest = divmod(time_taken, 3600)
minutes, seconds = divmod(rest, 60)

print ("Time: ", hours, "h, ", minutes, "min, ", seconds, "s ")
