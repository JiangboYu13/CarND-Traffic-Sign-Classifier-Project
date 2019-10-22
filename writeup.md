# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[dataset_example]: ./writeup_figure/dataset_example.png "Dataset Example Images"
[train_hist]: ./writeup_figure/train_hist.png "Train Image Histogram"
[valid_hist]: ./writeup_figure/valid_hist.png "Validation Image Histogram"
[test_hist]: ./writeup_figure/test_hist.png "Train Image Histogram"
[grayscale]: ./writeup_figure/grayscale.png "Grayscale"
[normalized]: ./writeup_figure/normalized.png "Normalized"
[augment]: ./writeup_figure/augment.png "Augmentated Data"
[web_figure]: ./writeup_figure/web_figure.png "Web Figure"
[web_his0]: ./writeup_figure/web_hist0.png
[web_his1]: ./writeup_figure/web_hist1.png
[web_his2]: ./writeup_figure/web_hist2.png
[web_his3]: ./writeup_figure/web_hist3.png
[web_his4]: ./writeup_figure/web_hist4.png
[web_his5]: ./writeup_figure/web_hist5.png

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JiangboYu13/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
![Example][dataset_example]
![Train][train_hist]
![Validation][valid_hist]
![Test][test_hist]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because:

My first implementation was using rgb scale image as input and used classroom LeNet-5 as network architecture. It gained a best validation set accuracy of 0.80, which was much worse than expected. Then I converted images to grayscale and without any other change, a best validation accuracy of 0.91 was seen. The reason of such a huge difference in performance might be: A rgb-scaled image contains much more features to learn, compared with gray scale one. Some of features might not be relavant to traffic sign recognation. By converting images to gray-scale, most of the features useful for recogination retains and the networks will be easier to train.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale][grayscale]

As a last step, I normalized the image data because this can change the values of input images in the dataset to a common scale, without distorting differences in the ranges of values, making the network more stable and easier to converge.
![normalized][normalized]
I decided to generate additional data because this can significantly increase the diversity of data available for training models, without actually collecting new data.  

To add more data to the the data set, I used the following techniques:
- Randomly change the brightness of images 
- Add gaussian noise to imagess

By changing teh brightness of images randomly to augment the training data, the network can be trained to address more essential features, making the network more robust to variance of brightness, which is the biggest challenge for computer vision.
By applying gaussian noise, again, it can make the network to address most essential features and ignore more subtle features.

Here is an example of an original image and an augmented image:

![augmentation][augment]

The difference between the original data set and the augmented data set is the following:
The augmented data set size is doubled and some noise is added to original data set.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten 				| output 400									|
| Fully connected		| output 120        							|
| dropout				| Keep Probabilty: 0.5        					|
| Fully connected		| output 84        							    |
| dropout				| Keep Probabilty: 0.5        					|
| Fully connected		| output 43        							    |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizor with training rate 0.0009. 
Unlike SGD which maintains a single learning rate for all weights update and such a rate will not change during training, the Adam optimizor adapts the per-parameter learning rates on the first and second moments of gradient.
the beta1, beta2 and epsilon just take their default value in tensorflow.
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.974
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
** The first model I tried was the classroom LeNet5 network and it gives me a validation accuracy of 91%.

* What were some problems with the initial architecture?
** The initial architecture didn't give me a validation accuracy better than 93%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

To achieve a better accuracy, I adopted two technique:
- Data augmentayion
  Change the brightness of original training data randomly and add Gaussian noise to it to produce more training data.
- Dropout
  Add dropout layer after the FC layer to overcome the overfitting issue.
  
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![web figure][web_figure] 

The second image (speed limit (20km/h) with label 0) might be difficult to classify because accourding to the training data historgram, there are too few training data for label 0.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing     | Children crossing  							| 
| Speed limit (20km/h)  | No passing for vehicles over 3.5 metric tons  |
| Double curve			| Double curve                                  |
| Roundabout mandatory	| Roundabout mandatory   		 				|
| No entry 				| No entry										|
| Keep left			    | Keep left	        							|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 95.2%


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
First image:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Children crossing   							| 
| .00     				| Right-of-way at the next intersection  		|
| .00					| Beware of ice/snow							|
| .00	      			| Bicycles crossing			 			    	|
| .00				    | Pedestrians       							|

![web histogram][web_his0] 
Second image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| No passing for vehicles over 3.5 metric tons	| 
| .24     				| Keep right                                	|
| .22					| Vehicles over 3.5 metric tons prohibited  	|
| .03	      			| Speed limit (50km/h)  	 			    	|
| .01				    | Speed limit (30km/h)							|

![web histogram][web_his1] 

Third image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .88         			| Double curve                              	| 
| .11     				| Right-of-way at the next intersection     	|
| .00					| Road narrows on the right                 	|
| .00	      			| Beware of ice/snow        			    	|
| .00				    | Bicycles crossing 							|

![web histogram][web_his2] 

Fourth image:

| Probability         	|     Prediction	        				    	| 
|:---------------------:|:-------------------------------------------------:| 
| 1.         			| Roundabout mandatory                          	| 
| .00     				| Speed limit (100km/h)     						|
| .00					| Priority road  					               	|
| .00	      			| No passing for vehicles over 3.5 metric tons  	|
| .00				    | End of no passing by vehicles over 3.5 metric tons|

![web histogram][web_his3] 

Fifth image:

| Probability         	|     Prediction	        				    	| 
|:---------------------:|:-------------------------------------------------:| 
| 1.         			| No entry                                         	| 
| .00     				| Stop     			                       			|
| .00					| Turn right ahead		        	               	|
| .00	      			| Go straight or left                           	|
| .00				    | Speed limit (120km/h)                             |

![web histogram][web_his4] 

Sixth image:

| Probability         	|     Prediction	        				    	| 
|:---------------------:|:-------------------------------------------------:| 
| .97         			| Keep left                                        	| 
| .02     				| Turn right ahead	                       			|
| .00					| Yield		        	                        	|
| .00	      			| Go straight or left                           	|
| .00				    | Speed limit (70km/h)                              |

![web histogram][web_his5] 



