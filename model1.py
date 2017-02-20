# source activate carnd-term1
# cd ~/CarND-Simulator
# python prepareListImages.py is a preprocessing to do an organization of
# the files

import numpy as np
import math
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import random
import csv
import cv2

from time import time
start_time = time()


tf.python.control_flow_ops = tf

print('Modules loaded.')
size1 = 64  # 160 * 0.4
size2 = 96  # 128 = 320 * 0.4


def random_shear(image, shear_range=200):
    """
    Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk

    :param image:
        Source image on which the shear operation will be applied


    :param shear_range:
        Random shear between [-shear_range, shear_range + 1] will be applied

    :return:
        The image generated by applying random shear on the source image
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle = dsteering

    return image, steering_angle


def generator(nbatch):

    # Open nameIdx.csv, a special file prepared with names and angles
    # nameIdx[0] = name of image  nameIdx[1] = angle with deviation left right
    lst_names = []
    fileName = 'Data/nameIdx.csv'

    i = 0

    # Open csv with names
    with open(fileName, "r") as oFile:
        wr = csv.writer(oFile, delimiter=',', quoting=csv.QUOTE_ALL)
        spamreader = csv.reader(oFile, delimiter=",")
        for row in spamreader:
            lst_names.append(row)

    #=============================
    # lstPicked: random picked images from dataset
    lstPicked = random.sample(lst_names, nbatch)
    # print(lst2)

    #=============================
    # Load images and angles from picked images
    imgScreenshots = []
    p = np.zeros(nbatch)
    folder = 'Data/IMG'

    low_threshold = 100
    high_threshold = 200
    i = 0
    for row in lstPicked:
        img = cv2.imread(os.path.join(folder, row[1]))  # 0 gray
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Random shear
        # if np.random.uniform() > 0.8:
        # 	img_shear, angle_shear = random_shear(img)

        # imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        #imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Just S channel
        # imgS = imgHSV[:,:,1].squeeze() #S Channel

        # Insertion of random lateral shift and equivalent compensation on angle
        # pix2angle = -0.05  #Opposed direction
        # latShift = random.randint(-10,10)
        # M = np.float32([[1,0,latShift],[0,1,0]])
        # imgTransladed = cv2.warpAffine(imgS,M,(imgS.shape[1],imgS.shape[0]))

        # crop_img = imgTransladed[40:160,10:310]

        # Full color image
        crop_img = imgRGB[40:160, 10:310, :]
        # crop_img = imgS[40:160,10:310]
        # crop_img = img_shear[40:160,10:310,:]

        # Insert value
        imgScreenshots.append(crop_img)

        p[i] = float(row[2])  # Value plus compensation
        # p[i]=float(row[2])+ angle_shear #Value plus shear
        # p[i]=float(row[2]) +latShift*pix2angle #Value plus compensation

        i += 1

    lenScreen = len(imgScreenshots)

    # X_in = np.zeros((lenScreen,size1,size2,1))
    X_in = np.zeros((lenScreen, size1, size2, 3))

    # Create np array with images
    for i in range(len(imgScreenshots)):
        X_in[i, :, :, :] = cv2.resize(
            imgScreenshots[i].squeeze(), (size2, size1))
        # X_in[i,:,:,0] =  cv2.resize(imgScreenshots[i].squeeze(), (size2,size1))

    # Create Y_in array
    Y_in = np.zeros((lenScreen))

    # Fill Y (label) values
    Y_in[0:lenScreen] = p[0:lenScreen]

    # Including flipped images
    for i in range(lenScreen):
        if np.random.uniform() > 0.5:
            X_in[i, :, :, :] = cv2.flip(X_in[i, :, :, :], 1)
            Y_in[i] = -p[i]  # Flipped images

    X_norm = X_in / 127.5 - 1
    Y_norm = Y_in / (1 + 0.01)  # a little gentler

    yield X_norm, Y_norm
    # return X_norm, Y_norm


#X_train, Y_train = generator(50)
# print(np.max(X_train[0].squeeze()))
# print(X_train[0].shape)

# for i in range(5):
# 	plt.subplot(5,1,i+1)
# 	plt.imshow(X_train[20+i].squeeze(), cmap='gray')
# plt.show()

# split_ratio =.95
# lenTr= int(split_ratio*len(X_norm))
# X_train, Y_train = X_shuff[0:lenTr], Y_shuff[0:lenTr]
# X_test, Y_test = X_shuff[lenTr:len(X_norm)], Y_shuff[lenTr:len(X_norm)]

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation, Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Model

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

model = Sequential()

# 0. Model chooses how to convert Color channels
model.add(Convolution2D(1, 1, 1, border_mode='same',
                        init='glorot_uniform', input_shape=(size1, size2, 3)))
# model.add(Convolution2D(3, 1, 1, border_mode='same'))


# 1. Convolutional, kernel(5,5) - Output 24
#model.add(Convolution2D(24,5,5, border_mode = 'valid', input_shape=(160,320,1)))
model.add(Convolution2D(24, 3, 3, border_mode='valid',
                        init='glorot_uniform', W_regularizer=l2(0.01)))

model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))
#
# 2. Convolutional, kernel(5,5) - Output 36
model.add(Convolution2D(36, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

# 3. Convolutional, kernel(5,5) - Output 48
model.add(Convolution2D(48, 3, 3, border_mode='valid', init='glorot_uniform'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))


# # ##4. Convolutional, kernel(3,3) - Output 64
# model.add(Convolution2D(64, 3,3, border_mode = 'valid'))
# model.add(Activation('elu'))

# 5. Flatten
model.add(Flatten())

# # #6. Dense 1164 in Nvidia End-to-End model
# model.add(Dropout(.5))
# model.add(Dense(500,init='uniform'))
# model.add(BatchNormalization())
# model.add(Activation('elu'))

# 7. Dense 100
model.add(Dropout(.5))
model.add(Dense(100, init='uniform'))
# model.add(BatchNormalization())
model.add(Activation('elu'))


# 7. Dense 50
# model.add(Dropout(.2))
model.add(Dense(100, init='uniform'))
# model.add(BatchNormalization())
model.add(Activation('elu'))


# 8. Dense 10
# model.add(Dropout(.2))
model.add(Dense(10, init='uniform'))
# model.add(BatchNormalization())
model.add(Activation('elu'))


# 9. Output
model.add(Dense(1))


LEARNING_RATE = 0.001
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mse')
#model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mae', metrics = ['accuracy'])

#filepath = "model_weights" + "-{epoch:02d}.h5"


n_epoch = 9
with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())

# To open previous model weights
# model.load_weights("model.h5")

for e in range(n_epoch):
    print("STAGE %d" % e)
    for X_train, Y_train in generator(8000):
        model.fit(X_train, Y_train, batch_size=200,
                  nb_epoch=1, verbose=1, validation_split=0.20)


# #Save model and weights
model.save('model.h5')
print ('model saved')
# with open('model.json', 'w') as outfile:
#     outfile.write(model.to_json())


#pred  = model.predict(X_test, batch_size=128)
#print("Test error: ", np.mean(np.square(pred-Y_test)))


end_time = time()
time_taken = end_time - start_time  # time_taken is in seconds

hours, rest = divmod(time_taken, 3600)
minutes, seconds = divmod(rest, 60)

print ("Time: ", hours, "h, ", minutes, "min, ", seconds, "s ")
