# Fashion-MNIST Clothing Apparel Image Classification using Deep Learning with Python
The Ultimate Guide for building Neural Networks to Classify Fashion Clothing Apparels using Keras and TensorFlow.

![image](https://user-images.githubusercontent.com/31254745/158236940-6e04c018-db77-4def-8d07-c2284446c4ce.png)


<p align="center">
    <p align="center" > Image Source: <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST samples </a> (by Zalando, MIT License).
</p>
  


## Steps to Build an Optimised Neural Network Image Classifier Model using Keras, TensorFlow & Hyperparameter Tuning with GridSearchCV.

1.	Problem Statement
2.	Dataset Description
3.	Importing Libraries
4.	Loading the Fashion MNIST dataset from Keras API
5.	Data Visualisation of Images in Training Data
6.	Data Pre-processing
7.	Model Building
8.	Model Evaluation
9.	Hyperparameter Tuning of the Neural Network Model using GridSearchCV
10.	Model Building: Neural Network with the Best Hyperparameters 
11.	Model Evaluation: Neural Network with the Best Hyperparameters 
12.	Data Visualisation of Neural Network Model Loss and Accuracy Results
13.	Predictions on Test Data

## 1.	Problem Statement

The objective of this task is we are given a Fashion-MNIST dataset available through Keras API. Using this data, we will build an optimised Image Classifier and demonstrate how we can harness the power of Deep Learning using Keras and TensorFlow.

In this blog, I will walk you through the entire process of implementing a feedforward neural network model on the Fashion-MNIST dataset to classify images of clothing apparel on train data and make predictions on test data using GridSearchCV Hyperparameter tuning technique to achieve the best accuracy and performance.

## 2.	Dataset Description

- The fashion-MNIST dataset consists of 60,000 training images and 10,000 test images of fashion product database images like Shirts, Bags, Sneakers etc.
- Fashion MNIST dataset can be accessed directly from Fashion MNIST TensorFlow using Keras API.
- Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
- Each pixel has a single pixel value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.

## 3.	Importing Libraries

We use Keras, a high-level API to build and train models in TensorFlow.

## 4.	Loading the Fashion MNIST dataset from Keras API

We use the Fashion MNIST dataset from Keras API using TensorFlow which contains 70,000 grayscale images. It has 10 different categories of fashion clothing apparel as shown below.

![image](https://user-images.githubusercontent.com/31254745/158238281-c3fe048b-3ff3-432c-8a49-f9d90c19327f.png)

After loading the dataset from Keras API, it returns four NumPy arrays.

- X_train and y_train arrays are the training set arrays used to train the neural network model on training data images.
- X_test and y_test arrays are the testing set arrays used to make predictions on the testing data images. 

## 5.	Data Exploration
 
- Number of observations in training data: 60000
- Number of labels in training data: 60000
- Dimensions of a single image in X_train: (28, 28)
- Number of observations in test data: 10000
- Number of labels in test data: 10000
- Dimensions of a single image in X_test: (28, 28)

## 6.	Data Visualisation of Images in Training Data

Plotting the first 25 images from the training set and displaying the class name below each image.

![image](https://user-images.githubusercontent.com/31254745/158238578-4ba50c2d-7e16-48b3-9607-c3180692db3a.png)

## 7.	Data Pre-processing
The data must be preprocessed before training the neural network model. If we check the first image in the training set, we will see that the pixel values fall in the range of 0 to 255.

Hence, we apply feature scaling to scale these values to a range of 0 to 1 before feeding them to the neural network model. So, divide the values by 255 and the training set and the testing set must be pre-processed in the same way.

## 8.	Model Building

Building the neural network model requires configuring the layers of the model and then compiling the model.

### 8.1 Neural Network Architecture

- The first layer in the network, tf. Keras. layers. Flatten, transforms the format of the images from a two-dimensional array of 28 x 28 pixels to a one-dimensional array of 28 x 28 = 784 pixels.
- After the pixels are flattened, the network consists of a sequence of two tf. Keras. layers. Dense layers. These are densely connected, or fully connected neural layers and it has 128 neurons.
- The last layer returns a logits array with a length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes/categories in the dataset.

### 8.2 Compile the Neural Network Model
- Loss function: It measures how accurate the model is during training. We want to minimize this function to move the model in the right direction.
- Optimizer: This is how the model is updated based on the data it sees and its loss function.
- Metrics: Used to monitor the training and testing steps. We use accuracy as an evaluation metric to check how accurately the images are classified.

### 8.3 Model Training
To start training, we use the model.fit method which is called because it "fits" the model to the training data and parameter with epochs = 50.

### 8.4 Model Evaluation
As the model training is completed, the loss and accuracy metrics are displayed.

**Results:**
- 1875/1875 - 2s - loss: 0.0884 - accuracy: 0.9678 - 2s/epoch - 1ms/step
- Training Accuracy: 96.78%
- 313/313 - 1s - loss: 0.4963 - accuracy: 0.8908 - 623ms/epoch - 2ms/step
- Testing Accuracy: 89.08%

We can observe that the accuracy of the Training dataset is 96.78% and the testing dataset is 89.08% which depicts the test dataset's accuracy is less than the accuracy on the training dataset. This gap between training accuracy and test accuracy is called Overfitting.

Overfitting happens when a deep learning model performs worse on new previously unseen data than it does on the training data.

Most of the deep learning models tend to be good at fitting to the training data, but the real challenge is Generalization, not fitting. Hence, to counter overfitting, we use different strategies and one of the techniques we use in this project is Hyperparameter tuning using GridSearchCV.

## 9.	Hyperparameter Tuning of the Neural Network Model using GridSearchCV

**Hyperparameter Tuning**

The process of selecting the right set of hyperparameters for the ML/DL model is called Hyperparameter tuning.
Hyperparameters are the variables that govern the training process and the topology of a model. These variables remain constant over the training process and directly impact the performance of the model.

**Grid Search**

Grid Search uses a different combination of all the specified hyperparameters and calculates the performance for each combination and selects the best value for the hyperparameters.

**Cross-Validation**

In GridSearchCV, along with Grid Search, cross-validation is also performed. Cross-Validation is used while training the model. As we know that before training the model with data, we divide the data into two parts – train data and test data.

In cross-validation, the process divides the train data further into two parts – the train data and the validation data.

The most popular type of Cross-validation is K-fold Cross-Validation. It is an iterative process that divides the train data into k partitions. Hence, Grid Search along with the cross-validation (GridSearchCV) technique takes huge time cumulatively to evaluate the best hyperparameters and build an optimised model.

### 9.1 Implementation of Hyperparameter Tuning of Neural Network Model using GridSearchCV
1.	Hyperparameter Tuning "Epochs"
2.	Hyperparameter Tuning "Batch Size"
3.	Hyperparameter Tuning "Learning Rate" and "Dropout Rate"
4.	Hyperparameter Tuning "Activation Function" and "Kernel Initializer"
5.	Hyperparameter Tuning "Hidden Layer Neuron 1" and "Hidden Layer Neuron 2"
6.	Hyperparameter Tuning "Optimizers"

The best hyperparameters obtained on the Fashion-MNIST dataset are shown below after GridSearchCV tuning.

![image](https://user-images.githubusercontent.com/31254745/158239734-7a722cf9-ac43-4382-874b-f8b1287a1b5d.png)

## 10.	Model Building with the Best Hyperparameters Obtained using GridSearchCV

After applying Hyperparameter tuning using GridSearchCV, we can observe that the accuracy of the Training dataset is 89.83% and the validation dataset is 86.96%.

In comparison with the neural network model accuracy obtained without hyperparameter tuning, there was a high gap between training and testing data but after applying the hyperparameter tuning GridSearchCV technique we were able to reduce the overfitting.

Hence, we can say that our Neural Network model with hyperparameter tuning is more generalized and prevented overfitting.

![image](https://user-images.githubusercontent.com/31254745/158240364-34d11df3-c1a9-4dd0-b222-d9c5da7d4e42.png)

 
**Results:**
- 1500/1500 - 2s - loss: 0.2915 - accuracy: 0.8983 - 2s/epoch - 2ms/step
- Training Accuracy: 89.83%
- 375/375 - 1s - loss: 0.3786 - accuracy: 0.8696 - 593ms/epoch - 2ms/step
- Validation Accuracy: 86.96%

## 11.	Neural Network Model Loss on Train and Validation Data

![image](https://user-images.githubusercontent.com/31254745/158240484-96284279-97a0-41af-b868-3599b633fae5.png)

## 12. Neural Network Model Accuracy on Train and Validation Data

![image](https://user-images.githubusercontent.com/31254745/158240564-c5adbd57-8efd-481c-be31-2b54fe42842c.png)

## 13. Predictions on Test Data

### 13.1	Overall Model Accuracy Results Summary

With the model trained with the best hyperparameters, we can use it to make predictions on test data. Accuracy given by the Training set is 89.83% and Accuracy given by the Testing set is 85.66%. Hence, we can say that the neural network model is more generalized (learns well) and even performs better on testing data.

![image](https://user-images.githubusercontent.com/31254745/158240688-b2b91f01-120a-4468-b158-a004606e0a77.png)

### 13.2	Model Evaluation: Classification Report on Test Data

![image](https://user-images.githubusercontent.com/31254745/158240865-00cc862c-f0d7-4315-bbf0-d553c243d068.png)

### 13.3	Plotting Predictions on Test Data Images

Plotting the first 25 test images with their predicted labels, and the true labels.

![image](https://user-images.githubusercontent.com/31254745/158240922-b2314cad-39a7-49fe-a15c-e624052e5c71.png)

## 14.	Conclusion
In this blog, we discussed how to approach the image classification problem by implementing a Neural networks model using Keras, TensorFlow and GridSearchCV. 

We can explore this work further by trying to improve the accuracy by using advanced Deep Learning algorithms like Convolutional Neural Networks (CNN).

## 15.	References
- [1] https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
- [2] https://www.tensorflow.org/tutorials/keras/classification
- [3] https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data





