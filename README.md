# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center.png "Center"
[image2]: ./images/left.png "Left"
[image3]: ./images/right.png "Right"
[image4]: ./images/flipped.png "Flipped"
[image5]: ./images/epoch5.png "Epoch 5"
[image6]: ./images/epoch2.png "Epoch 2"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
To train the model:
```sh
python model.py
```
To run the simulator with the weight:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used an modified version of Nvidia architecture. The model consists of the following architecture. (model.py line 68-84)
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							|
| Normalization         		|    							| 
| Cropping         		|   							| 
| Convolution 5x5     	| 2x2 stride, valid padding	  |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding	  |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding	  |
| RELU					|												|
| Convolution 3x3     	| 2x2 stride, valid padding	  |
| RELU					|												|
| Flatten            |       |
| Fully connected | outputs 100|
| Fully connected | outputs 50|
| Fully connected | outputs 10|
| Fully connected | outputs 1|


#### 2. Attempts to reduce overfitting in the model

To combat overfitting, I split the data into 80% training and 20% validation data. During training, the training data is shuffled in generator. 
During the training process, the mse error of validation error got higher than the training, so I lower the epoch to obtain the optimal model. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with correction value of 0.2. In addition, to generalize the data better, I flipeed the images so it has a balanced left and right turn view. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the existing architecture and test it on our simulation data. 

For the existing architecture I used the Nivida architecture which is appropriate since it was also used for self-driving cars. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set after the second epoch, which implies that it was overfitting. 

To combat the overfitting, I modified the model to run only two epoch to obtain the best weight values.

The final step was to run the simulator to see how well the car was driving around track one. Interestingly, the car seems to drive around in circles which indicates there is something wrong with my model.

Then I switch the activation back to "relu" and the simulation got a lot better. However, there are still a few spots that the car would fell off the track, such as the turn before the bridge and after the bridge. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-84) was the Nvidia architecture with slight modification. 
The visualization of the architecture is below.

#### 3. Creation of the Training Set & Training Process

Due to the lack of memory space on my computer, I simply used Udacity's sample data for training. Here is an example image of center, left, and right lane driving:

![alt text][image1]
![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would generalize the dataset so left and right turn are balanced. For example, here is an image that has then been flipped:

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the graph below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Training with Epoch of 5:
![alt text][image5]
Training with Epoch of 2:
![alt text][image6]
