# Flower Image Classifier
Image classifier for classifying flower types using a pre-trained neural network (either vgg19 or alexnet). 

# Data
The dataset used to train the neural network is the 102 Category Flower Dataset. It contains 102 different categories of flowers. Each category contains between 40 and 258 images. 
More information about the dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

# Files 
This classifier uses four .py files and a .json file:
* model_settings.py  
  This file defines the functions used in the main train.py and predict.py files
* utility.py  
This file defines the files for importing and preprocessing data prior to training. 
* train.py  
This is the training file. It requires the directory containing the images as an input argument. It has optional arguments including: the neural network model (vgg19, alexnet), the learning rate, the hidden layer size, number of training epochs and the option of running it on the cpu or a gpu. 
This script also reports the classification accuracy of the model on the training set. 
* Predict  
This file takes an input of an image of a flower and predicts the category of the flower. It requires the path to the image as an argument, the checkpoint from the trained model, the number of top categories to report, the device on which to run the script (cpu or gpu) and the file containing the labels. 
* cat_to_name.json   
This files contains the information to convert the category number to the category name. 
