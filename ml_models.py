#Basic imports and plot packages
import numpy as np
import os
from PIL import Image

#Machine learning packages
from sklearn.model_selection import train_test_split


#Importing the data images
def import_images(path):
    return os.listdir(path)


#Convert images to array 
def convert_to_array(files, path):
    image_data = []

    #Iterating over every file
    for i in range(len(files)):
        #Opening the Image and converting it to GrayScale image using .convert('L')
        img = Image.open(f'{path}/{files[i]}').convert('L')

        #Converting the image to a numpy array
        img = np.asarray(img)

        #Flattening the image
        img = np.ravel(img.reshape(1, img.shape[0]*img.shape[1])).tolist()

        #Storing the flattened image to a list
        image_data.append(img)

    return equate_shape(image_data=image_data)


#Equate shape of all the image data
def equate_shape(image_data):
    #There are some images that are not of the same shape, so we need to make their length same

    #Finding the length of the first image file
    length = len(image_data[0])

    #Iterating over every image
    for i in range(len(image_data)):

        #Finding the margin by which the length falls short
        size = int(length - len(image_data[i]))

        #Creating a list of that many number of arrays
        zeros = [0]*size    

        #Checking if the length image data is short
        if len(image_data[i]) < length:

            #Appending that many number of zeros at the end of that image data to make it of same size
            image_data[i].extend(zeros)

    #Converting the list to numpy array
    image_data = np.array(image_data)
    return image_data


#Spliting the data into train and test set
def split_data(image_data, labels):
    return train_test_split(image_data, labels, test_size=0.2, random_state=123)



def get_shape(path, files):
    #Opening the Image and converting it to GrayScale image using .convert('L')
    img = Image.open(f'{path}/{files[0]}').convert('L')

        #Converting the image to a numpy array
    img = np.asarray(img)

    return img.shape