#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/Center.png "Center Driving "
[image2]: ./Images/recovery.png "recovery"
[image3]: ./Images/bridgerecovery.png "Recovery Image"
[image4]: ./Images/alldata.png "all data"
[image5]: ./Images/filtereddata.png "all data"
[image6]: ./Images/nVidia_model.png "all data"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* WorkingModel-Final.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* mymodel9.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The WorkingModel-Final.py  file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
I have used the Basic NVidia model with no changes . I just tweaked the number of epochs and samples per epoch.
Here is what the basic model looks like
![alt text][image6]



####2. Attempts to reduce overfitting in the model

The NVidia model has dropout layers before each Fully connected layer. I just ensured that the number of epochs were correctly set so as to not overtrain the network

####3. Model parameter tuning

The model used an adam optimizer with a learning rate of 1e-4.

####4. Appropriate training data

This was the main part of the project as far as i could see. I adopted the following strategy:
* First i simply generated data for 1 lap and ensured that the the car still drives with a basic model( and ofcourse crashes)
* Once i was sure the system is in place i collected data for 3 laps of center driving
* I used this data along with the Udacity Data to generate the training data set. And split this combination to get the validation set. I used data from all three cameras with a correction angle ( which i reached on by trial and error) so i got a pretty large data set
* Next i created some recovery data explained in another section below. For each set of recovery data i created 10 sets of data ( ln 127-194). I just wanted to create more examples of recovery. For details about how I created the training data, see the next section. 
* I also avoided adding images with 0 speed as they were not representative of driving

###Model Architecture and Training Strategy

####1. Solution Design Approach
After reading a few blogs about this project i noticed that most students were sticking with nVidia or Comma.ai. I decided to try these out to start with and nVidia worked for me without any modifications

####2. Final Model Architecture

The final model architecture  consisted of a nVidia model neural network without any modifications


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it reaches an edge This image is an example of what a recovery looks like starting from ... :

![alt text][image2]

I was then having issues on the bridge and recored some data for recovery on the bridge.

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would help get the network more examples of left and right steering. This was done in the generator(ln 376)

After the collection process, I had 71911 number of data points. I then preprocessed this data by :
* Cropping the sky and bonnet section off
* Resizing to 64*64 as needed by nVidia architecture
* Converting to YUV scale as recommended by nVidia

The same Preprocessing steps were added to drive.py also

####4.Data Distribution Flattening
(ln 398 to 435)
I also used tried to adjust the number of images i had per steering angle so as to reduce any bias. I used a histogram to plot number of images per angle and then reduced the ones with more data.
Here is what the histogram looked like before flattening

![alt text][image4]

This made sense since i had added all 3 camera images with a correction of +-.25
Post flattening the data was more 'equally distributed'

![alt text][image5]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

####5. Use of Generators
I used a Python generator to generate data for each batch of the training and validation set. This helped prevent memory errors. I ensured the generator was shuffling the images before each iteration. The generator would also create a flipped image (ln 376-384) and augment the data set.

####5. Driving the car!

With the groundwork in place i experimented with number of epochs and samples per epochs . Once i saw the loss getting reduced i tested the model and the car was able to drive by without any crashes. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the validation loss. I used an adam optimizer with a learning rate of 1e-4( reached here after a few experiments)

The car crashes on track 2 because for that we should ideally augment the data with shadows and some distortion. Also i did not train the car on track 2 at all ( due to lack of time).
