import os
import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sklearn.metrics import f1_score

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

class GetLicensePlateNumber:
    def __init__(self,imgPath='vehicle1.jpg'):
        # Read the image file
        self.image = cv2.imread(imgPath)
        t=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.imshow(t)
        plt.title('Original Image')
        plt.show()
        self.image = imutils.resize(self.image, width=500)
        self.img=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Display the original image
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10,7))
        self.ax[0,0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.ax[0,0].set_title('Original Image')
    def img_processing(self,):
        # RGB to Gray scale conversion
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.ax[0,1].imshow(gray, cmap='gray')
        self.ax[0,1].set_title('Grayscale Conversion')

        blur = cv2.GaussianBlur(gray, (3,3), 0)

        # Noise removal with iterative bilateral filter(removes noise while preserving edges)
        gray = cv2.bilateralFilter(blur, 11, 17, 17)
        self.ax[1,0].imshow(gray, cmap='gray')
        self.ax[1,0].set_title('Bilateral Filter')

        # Find Edges of the grayscale image
        self.edged = cv2.Canny(gray, 170, 200)
        self.ax[1,1].imshow(self.edged, cmap='gray')
        self.ax[1,1].set_title('Canny Edges')

        self.fig.tight_layout()
        plt.show()
    def find_contours(self,):
        # Find contours based on Edges
        self.cnts = cv2.findContours(self.edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        self.cnts=sorted(self.cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
        self.NumberPlateCnt = None #we currently have no Number plate contour
        # loop over our contours to find the best possible approximate contour of number plate
        count = 0
        for c in self.cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:  # Select the contour with 4 corners
                    self.NumberPlateCnt = approx #This is our approx Number Plate Contour
                    x,y,w,h = cv2.boundingRect(c)
                    self.ROI = self.img[y:y+h, x:x+w]
                    break

        if self.NumberPlateCnt is not None:
            # Drawing the selected contour on the original image
            cv2.drawContours(self.image, [self.NumberPlateCnt], -1, (0,255,0), 3)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Detected license plate")
        plt.show()
        #The green bounding box shows the detected license plate.
        # Find bounding box and extract ROI
        plt.imshow(self.ROI)
        plt.title("Extracted license plate")
        plt.show()
        #The above displayed coordinates are the coordinates of the detected plate. But the problem is that we don't know which coordinate is where, because contours can start from anywhere and form a continuous path.

    #The idea behind plate rotation is to find the bottom two coordinates. Using these two coordinates, we can easily find the angle of rotation. This will be illustrated shortly.
    # Distance between (x1, y1) and (x2, y2)
    def dist(self,x1, x2, y1, y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5
    def straighten_licenseplate(self,):
        """The above function returns the Euclidean distance between any two points (x1, y1) and (x2, y2).
        As discussed, we need to find the bottom two coordinates:
        - To find them, we'll first find a coordinate with the maximum y-coordinate and this will be one of the two bottom-most coordinates.
        - Now, the other bottom coordinate will be either to the left or right of this coordinate in the array. Since, license plates are rectangular in shape, the second required coordinate would be in a distance far away from the acquired coordinate than the other adjacent coordinate."""
        idx=0
        m=0
        # To find the index of coordinate with maximum y-coordinate
        for i in range(4):
            if self.NumberPlateCnt[i][0][1]>m:
                idx=i
                m=self.NumberPlateCnt[i][0][1]

        # Assign index to the previous coordinate
        if idx==0:
            pin=3
        else:
            pin=idx-1

        # Assign index to the next coordinate
        if idx==3:
            nin=0
        else:
            nin=idx+1

        # Find self.distances between the acquired coordinate and its previous and next coordinate
        p=self.dist(self.NumberPlateCnt[idx][0][0], self.NumberPlateCnt[pin][0][0], self.NumberPlateCnt[idx][0][1], self.NumberPlateCnt[pin][0][1])
        n=self.dist(self.NumberPlateCnt[idx][0][0], self.NumberPlateCnt[nin][0][0], self.NumberPlateCnt[idx][0][1], self.NumberPlateCnt[nin][0][1])

        # The coordinate that has more self.distance from the acquired coordinate is the required second bottom-most coordinate
        if p>n:
            if self.NumberPlateCnt[pin][0][0]<self.NumberPlateCnt[idx][0][0]:
                left=pin
                right=idx
            else:
                left=idx
                right=pin
            d=p
        else:
            if self.NumberPlateCnt[nin][0][0]<self.NumberPlateCnt[idx][0][0]:
                left=nin
                right=idx
            else:
                left=idx
                right=nin
            d=n
        #Extract the coordinates of the bottom-most coordinates in such a way that ```(left_x, left_y)``` denote the bottom-left coordinate and ```(right_x, right_y)``` denote the bottom-right coordinate.
        left_x=self.NumberPlateCnt[left][0][0]
        left_y=self.NumberPlateCnt[left][0][1]
        right_x=self.NumberPlateCnt[right][0][0]
        right_y=self.NumberPlateCnt[right][0][1]
        #For rotating the plate, we need to find the angle of rotation. This can be found out by calculating the sin of theta using the two coordinates. Theta can then be extracted by finding the inverse of sin. Image can finally be rotated by using ```cv2.getRotationMatrix2D()``` function.
        # Finding the angle of rotation by calculating sin of theta
        opp=right_y-left_y
        hyp=((left_x-right_x)**2+(left_y-right_y)**2)**0.5
        sin=opp/hyp
        theta=math.asin(sin)*57.2958

        # Rotate the image according to the angle of rotation obtained
        image_center = tuple(np.array(self.ROI.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
        result = cv2.warpAffine(self.ROI, rot_mat, self.ROI.shape[1::-1], flags=cv2.INTER_LINEAR)

        # The image can be cropped after rotation( since rotated image takes much more height)
        if opp>0:
            h=result.shape[0]-opp//2
        else:
            h=result.shape[0]+opp//2

        self.result=result[0:h, :]
        plt.imshow(self.result)
        plt.title("Plate obtained after rotation")
        plt.show()
        return self.result
    ## Character Segmentation
    #Character segmentation is an operation that seeks to decompose an image of a sequence of characters into subimages of individual symbols. It is one of the decision processes in a system for optical character recognition (OCR).

    #This phase contains the use of two functions: ```segment_characters()``` and ```find_contours()```.
    # Match contours to license plate or character template
    def find_contours_of_numbers(self,dimensions, img) :

        # Find all contours in the image
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Retrieve potential dimensions
        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]
        
        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
        ii = cv2.imread('contour.jpg')
        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs :
            # detects contour in binary image and returns the coordinates of rectangle enclosing it
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            
            # checking the dimensions of the contour to filter out the characters by contour's size
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

                char_copy = np.zeros((44,24))
                # extracting each character using the enclosing rectangle's coordinates.
                char = img[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))
                
                cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
                plt.imshow(ii, cmap='gray')
                plt.title('Predict Segments')

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy) # List that stores the character's binary image (unsorted)
        # Return characters on ascending order with respect to the x-coordinate (most-left character first)
                
        plt.show()
        # arbitrary function that stores sorted list of character indeces
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])# stores character images according to their index
        img_res = np.array(img_res_copy)
        return img_res
    """In the above function, we will be applying some more image processing to extract the individual characters from the license plate. The steps involved will be:
    - Finding all the contours in the input image. The function cv2.findContours returns all the contours it finds in the image.
    - After finding all the contours we consider them one by one and calculate the dimension of their respective bounding rectangle. Now consider bounding rectangle is the smallest rectangle possible that contains the contour. All we need to do is do some parameter tuning and filter out the required rectangle containing required characters. For this, we will be performing some dimension comparison by accepting only those rectangle that have:
    1. Width in the range 0, (length of the pic)/(number of characters) and,
    2. Length in a range of (width of the pic)/2, 4*(width of the pic)/5. After this step, we should have all the characters extracted as binary images."""
    # Find characters in the resulting images
    def segment_characters(self,image) :

        # Preprocess cropped license plate image
        img_lp = cv2.resize(image, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        # Make borders white
        img_binary_lp[0:3,:] = 255
        img_binary_lp[:,0:3] = 255
        img_binary_lp[72:75,:] = 255
        img_binary_lp[:,330:333] = 255

        # Estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH/6,
                        LP_WIDTH/2,
                        LP_HEIGHT/10,
                        2*LP_HEIGHT/3]
        plt.imshow(img_binary_lp, cmap='gray')
        plt.title('Contour')
        plt.show()
        cv2.imwrite('contour.jpg',img_binary_lp)

        # Get contours within cropped license plate
        char_list = self.find_contours_of_numbers(dimensions, img_binary_lp)

        return char_list
    def get_img_numbers(self,char):
        self.new_char=[]
        for x in range(len(char)):
            if(not 255 in char[x][2]):
                self.new_char.append(char[x])
        for i in range(len(self.new_char)):
            plt.subplot(1, len(self.new_char), i+1)
            plt.imshow(self.new_char[i], cmap='gray')
            plt.axis('off')
        plt.show()
    """The above function takes in the image as input and performs the following operation on it:
    - Resizes it to a dimension such that all characters seem distinct and clear.
    - Convert the colored image to a gray scaled image. We do this to prepare the image for the next process.
    - Now the threshold function converts the grey scaled image to a binary image i.e each pixel will now have a value of 0 or 1 where 0 corresponds to black and 1 corresponds to white. It is done by applying a threshold that has a value between 0 and 255, here the value is 200 which means in the grayscaled image for pixels having a value above 200, in the new binary image that pixel will be given a value of 1. And for pixels having value below 200, in the new binary image that pixel will be given a value of 0.
    - The image is now in binary form and ready for the next process Eroding. Eroding is a simple process used for removing unwanted pixels from the object’s boundary meaning pixels that should have a value of 0 but are having a value of 1.
    - The image is now clean and free of boundary noise, we will now dilate the image to fill up the absent pixels meaning pixels that should have a value of 1 but are having value 0.
    - The next step now is to make the boundaries of the image white. This is to remove any out of the frame pixel in case it is present.
    - Next, we define a list of dimensions that contains 4 values with which we’ll be comparing the character’s dimensions for filtering out the required characters.
    - Through the above processes, we have reduced our image to a processed binary image and we are ready to pass this image for character extraction."""
    def load_saved_weights(self):
        """Since the data is all clean and ready, now it’s time do create a Neural Network that will be intelligent enough to recognize the characters after training. In this project, we used CNN model for character recognition.
        - For training the model, we’ll be using ImageDataGenerator class available in keras to generate some more data using image augmentation techniques like width shift, height shift.
        - Width shift: Accepts a float value denoting by what fraction the image will be shifted left and right.
        - Height shift: Accepts a float value denoting by what fraction the image will be shifted up and down.
        For the model, we'll use 4 convolutional layers with a Max pooling layer of window size = (4,4). We'll also use 2 Dense layers where the last dense layers will have 36 output units (26 alphabets + 10 digits) and the activation function used will be 'softmax' because this is a multi-classification problem.

        The below parameters used in the model have already been optimized using hyperparamter tuning.
        We'll now go ahead and test our model. Note that the attribute ```steps_per_epoch``` is set to be ```train_generator.samples // batch_size``` because it ensures the usage of all the train data for one epoch.
        Since we saved only the model weights, we first need to create a model instance and then load the saved weights into the model."""
        # Create a new model instance
        self.loaded_model = Sequential()
        self.loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
        self.loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
        self.loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
        self.loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
        self.loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
        self.loaded_model.add(Dropout(0.4))
        self.loaded_model.add(Flatten())
        self.loaded_model.add(Dense(128, activation='relu'))
        self.loaded_model.add(Dense(36, activation='softmax'))
        # Restore the weights
        self.loaded_model.load_weights('checkpoints/my_checkpoint')

# Predict the license plate number
#We now have our license plate and the CNN model ready! We just need to predict each character using the model. For this, we'll first fix the dimension of each character image using the function ```fix_dimension```, in which it converts an image to a 3-channel image. The image can then be sent to ```model.predict_classes()``` in order to get the predicted character.
# Predicting the output
    def fix_dimension(self,img): 
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
            return new_img
    
    def show_results(self,):
        dic = {}
        characters = '0123456789'
        for i,c in enumerate(characters):
            dic[i] = c

        output = []
        for i,ch in enumerate(self.new_char): #iterating over the characters
            img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            img = self.fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_ = np.argmax(self.loaded_model.predict(img)[0], axis=-1) #predicting the class
            character = dic[y_]
            output.append(character) #storing the result in a list
            

        plate_number = ''.join(output)

        return plate_number
    def predict_numbers_value(self):
        plate_number=self.show_results()
        print(self.show_results())
        # Segmented characters and their predicted value.
        plt.figure(figsize=(10,6))
        for i,ch in enumerate(self.new_char):
            img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
            plt.subplot(3,4,i+1)
            plt.imshow(img,cmap='gray')
            plt.title(f'predicted: {self.show_results()[i]}')
            plt.axis('off')
        plt.show()
licenseplateNumber=GetLicensePlateNumber()
licenseplateNumber.img_processing()
licenseplateNumber.find_contours()
result = licenseplateNumber.straighten_licenseplate()
char=licenseplateNumber.segment_characters(result)
licenseplateNumber.get_img_numbers(char)
licenseplateNumber.load_saved_weights()
licenseplateNumber.predict_numbers_value()