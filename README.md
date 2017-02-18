#**Behavioral-Cloning** 
###Teaching a car to drive himself

#PROBLEM !!!
 The car keeps driving off the road to the dirt road right after the bridge. I think the problem has to do with the image preprocessing or with the differnet image formats in drive.py/model.py. Please shed some light on this and point me in the right direction.
 I added a lot of training data only of the section where the car goes off the road but it has absolutely no effect.

#1. Objective 


In this Project we were asked to design a system that drives a car autonomously in a simulated environment.


Illustration of the simulator
![alt text][image1]


We are provided with the Udacity self-driving car simulator.This way we can pilto a car in a virtual environment and record the data.The simulator records screen shots of the images and the decisions we made: steering angles, throttle and brake. In this project we will focus only on the steering angle.

The ultimate goal is to create a deep neural network to emulate the behavior of the human being, and the put this network to run the simulator autonomously.


[//]: # (Image References)

[image1]: simulator.png "Simulator"
[image2]: model.png "Model"
[image3]: train_stats.png "Training stats"


#2. Strategy: 
My approach is based on the [NVIDIA End-to-End Slouttion](https://arxiv.org/abs/1604.07316)
I trained a convolutional neural network (CNN) to map raw pixels from the simulator directly to steering commands. 
The system should automatically learn internal representations of the necessary processing
steps such as detecting useful road features with only the human steering angle
as the training signal. We never explicitly train it to detect, for example, the outline
of roads.


#3.Model Architecture 

![alt text][image2]

I was able to simplify the NVIDIA End-to-End Slouttion and drastically reduce the number of parameters. I use less fully connected layers and also less convolution layers.

Our network architecture is shown in the Figure above. The network consists of 8 layers, including a 1x1 convolution at the input layer, three convolutional layers and three fully connected layers. 

The convolutional layers were designed to perform feature extraction and were chosen empirically
through a series of experiments that varied layer configurations. In the first two convolutional layers I used non strided convolution with a 5×5 kernel. In the last convolutional layer I used kernel size 3×3.

We follow the convolutional layers with three fully connected layers leading to an output control
value which is the normalized steering angle. The fully connected layers are designed to function as a
controller for steering, but we note that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.

The model includes ELU layers to introduce nonlinearity.


#4. Attempts to reduce overfitting in the model

The model contains a dropout layer after the flatten layer in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#5. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).
I implemented early termination to get the correct number of epochs.

![alt text][image3]


#6. Training Details

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

The most challenging part was to teach the car not to drive off the road in the section after the bridge, where lane lines are not marked clearly.
I followed my reviewer's recommendation and did the following to solve the problem:
I drove to the position where the car drives off to the dirt road and stopped it at the same position with  similar orientation.
Than I turned the wheel toward the center of the road  and recorded the data while the car standing for few seconds (30s).
This way our model learned what to do when it encounters a dirt road.


#7. Image augmentation

##7.1 Center and lateral images

I added a correction angle of 0.10 to the left image, and -0.10 to the right one. The idea is to center the car, avoid the borders.
 
 
##7.2 Flip images
 
 We can randomly choose to flip the image, and invert the steering angle. This way, we can neutralize some tendency of the human driver that drove a bit more to the left or to the right of the lane.
 

##7.3 Resize

Because of computational limits, it is a good thing to resize the images, after cropping it. I scaled the images down to half of there size

 
##7.4 Crop 
The image was cropped to remove irrelevant portions of the image, like the sky, or the trees. 


##7.5 Normalization
Normalization is made because the neural network usually works with small numbers.



 