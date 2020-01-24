## **Self-Driving Car Engineer Nanodegree Program**

## **Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/mean_square_error_loss.png "Error Loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** summarizing the results

##### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

##### 1. An appropriate model architecture has been employed

While going through my final solution I progressed by starting with initial basic model. My first model was very simple, I designed this one without thinking any tunning alternatives or would my training dataset enough to prepare well working algorithm etc. I just want to create a model would be used as benchmark model.

As a second solution I decided to go with old and well known LeNet, and added some normalization steps but it didn't end well, while I could not able to complete track1.

And as a final step I tried to go with Nvidia Autonomous Car network model. And car succesfully completed the Track1 by using `model.h5` file that my algorithm created just with three training epochs.

Final model (Nvidia Autonomous Car network model released in paper) architecture can be seen below. I just added 1 fully connected layer to the top, while we need 1 output, and one dropout layer after convolutional layer to prevent overfitting.

![alt text][image1]

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

##### 2. Attempts to reduce overfitting in the model

I have only 3 epochs which would not cause overfitting but also I added just one dropout layer to prevent overfitting. And also I split my sample data into training and validation data by 80% and 20%, to ensure that the model was not overfitting.

##### 3. Model parameter tuning

The model used an adam optimizer, so that manually training the learning rate wasn't necessary.

##### 4. Appropriate training data

I used the three different view (center, right, left) training data that provided by Udacity for both track.

### Model Architecture and Training Strategy

##### 1. Solution Design Approach

My first step was starting with very simple model, to create a benchmark. But the neural network outputs were very unsatisfactory, and by using this algorithm `model.h5` file, vehicle can go on road at all. Then I continued with LeNet by adding normalization and shuffling train and test data to avoid over-fitting. I cannot say I've totally satisfied with this network outputs but it was far more better that the first one, and before went out to the road I passed the bridge with out hitting somewhere. As a final step, I decided to go with more power-full neural network model which is Nvidia Autonomous Car model. I just added 1 fully connected layer to the top, while we need 1 output, and one dropout layer after convolutional layer to prevent overfitting. And also to increase the performance, addition the normalization that I already applied, I used center, right and left camera view combined them and I added some augmented data by flipping existing ones. But with this solution, my trained network failed to turn last left turn on the first track, therefore as a final rework I added one dropout layer and a elimination code which remove steering measurement data which is smaller than, 0.001. But I again didn't satisfied with the solution. Therefore I commented which remove steering wheel code and run it again. And finally I satisfied with this trained neural network. Which was used to create final mp4 file for my project.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

##### 2. Final Model Architecture

You can see the final model architecture:

```
Layer (type)                     Output Shape          Param #     Connected to           
=========================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]   
_________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 100, 320, 3)   0           lambda_1[0][0]       
_________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]     
_________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0] 
_________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0] 
_________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0] 
_________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0] 
_________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 13, 64)     0           convolution2d_5[0][0] 
_________________________________________________________________________________________
flatten_1 (Flatten)              (None, 832)           0           dropout_1[0][0] 
_________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]       
_________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]         
_________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]         
_________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]         
=========================================================================================
```

##### 3. Creation of the Training Set & Training Process

I used given training data given by Udacity, somehow I really frustrated to create training data by driving car in the simulation platform, I'm not sure why, but it would related with connection problem or related with my computer or network setting. Because I also face with similar problem, while driving car in the autonomous mode. But to pass over this problem, I played with the given data to increase performance of the neural networks that I applied. Below you can see my application for all training data:

* Right and left camera view combined
* Added augmented data by flipping existing camera view
* Train/Test split to balance data

All these data was used for training the model with three epochs. The data was shuffled randomly. The following picture shows the training:

Total Images: 24108
Train samples: 19286
Validation samples: 4822

19286/19286 [==============================] - 1972s 102ms/step - loss: 0.0078 - val_loss: 0.0105
Epoch 2/3
19286/19286 [==============================] - 1976s 102ms/step - loss: 0.0022 - val_loss: 0.0111
Epoch 3/3
19286/19286 [==============================] - 2364s 123ms/step - loss: 0.0012 - val_loss: 0.0110
dict_keys(['loss', 'val_loss'])
**Loss**
[0.007781327234576106, 0.002220115302492939, 0.001221665536940149]
**Validation Loss**
[0.01047553695908839, 0.01112232083437769, 0.011014808510371467]