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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

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

I used the Nvidia architecture as suggested in the guidance. The model consists of the following architecture. (model.py line )
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding	  |
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding	  |
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding	  |
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding	  |
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding	  |
| RELU					|												|
| Flatten            | outputs 1200  |
| Fully connected | outputs 100|
| Fully connected | outputs 50|
| Fully connected | outputs 10|
| Fully connected | outputs 1|


#### 2. Attempts to reduce overfitting in the model

To combat overfitting, I split the data into 80% training and 20% validation data. During training, the data is shuffled before yielding in generator. 
During the training process, the mse error of validation error got higher than the training, so I lower the epoch to obtain the optimal model. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with correction value of 0.2. In addition, to generalize the data better, I flipeed the images so it has a balanced left and right turn view. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the existing architecture and test it on our simulation data. 

For the existing architecture I used the Nivida architecture which is appropriate since it was also used for self-driving cars. 

Initially, instead of using "relu" as my activation function in my convolution layer, I used "elu". 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set after the second epoch, which implies that it was overfitting. 

To combat the overfitting, I modified the model to run only two epoch to obtain the best weight values.

The final step was to run the simulator to see how well the car was driving around track one. Interestingly, the car seems to drive around in circles which indicates there is something wrong with my model.

Then I switch the activation back to "relu" and the simulation got a lot better. However, there are still a few spots that the car would fell off the track, such as the turn before the bridge and after the bridge. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
