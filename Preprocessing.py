#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import math

train_labels = pd.read_csv('./train_labels.csv')
train_images = pd.read_pickle('./train_images.pkl')
test_images = pd.read_pickle('test_images.pkl')


# plt.title('Label: {}'.format(train_labels.iloc[16]['Category']))
# plt.imshow(train_images[16])
# plt.show()

""" Preprocessing using OpenCV : Choose between largest rectangle, ellipse, or contour """

def preprocessTrainMinRect():
	for i, img in enumerate(train_images):
		img = train_images[i].astype('uint8')

		#Threshold the image for values of 0 or 1
		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
		# plt.imshow(thresh)
		# plt.show()

		#Get contours of the binary image
		# contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
        # I think I'm using a diff openCV version where the return statment has
        # changed
		ctrs, _ = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		#Get the area of the largest rectangle
		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			temparea = cv2.minAreaRect(ctr)[1][0]*cv2.minAreaRect(ctr)[1][1] #length x width
			if (temparea>area):
				largestctr = ctr
				area = temparea

		#Make a picture of the image with only one (largest) number
		rect = cv2.minAreaRect(largestctr)
		box = cv2.boxPoints(rect) #get the four corners
		box = np.int0(box)
		poly = np.array([box], dtype=np.int32)
		mask = np.zeros(img.shape, np.uint8)
		cv2.fillPoly(mask,poly,255) #creating a mask over the irrelevant number contours

		newimg = cv2.bitwise_and(thresh, mask) #black and white image containing only the largest number
		xtrain.append(newimg)

def preprocessTestMinRect():
	for i, img in enumerate(test_images):
		img = test_images[i].astype('uint8')

		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

		# contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
        # I think I'm using a diff openCV version where the return statment has
        # changed
		ctrs, _ = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			temparea = cv2.minAreaRect(ctr)[1][0]*cv2.minAreaRect(ctr)[1][1]
			if (temparea>area):
				largestctr = ctr
				area = temparea

		rect = cv2.minAreaRect(largestctr)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		poly = np.array([box], dtype=np.int32)
		mask = np.zeros(img.shape, np.uint8)
		cv2.fillPoly(mask,poly,255)

		newimg = cv2.bitwise_and(thresh, mask)
		xtest.append(newimg)

def preprocessTrainMinEllipse():
	for i, img in enumerate(train_images):
		img = train_images[i].astype('uint8')

		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

		# contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
        # I think I'm using a diff openCV version where the return statment has
        # changed
		ctrs, _ = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		# Get contour of largest ellipse size
		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			if len(ctr) >= 5:
				(x,y), (MA, ma), angle = cv2.fitEllipse(ctr)
				temparea = (math.pi)*MA*ma
				if(temparea>area):
					largestctr = ctr
					area = temparea
			

		ellipse = cv2.fitEllipse(largestctr)
		mask = np.zeros(img.shape, np.uint8)
		poly = cv2.ellipse(mask, ellipse, (255,255,255),-1)

		newimg = cv2.bitwise_and(thresh, thresh, mask=poly)
		xtrain.append(newimg)

def preprocessTestMinEllipse():
	for i, img in enumerate(test_images):
		img = test_images[i].astype('uint8')

		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

		# contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
        # I think I'm using a diff openCV version where the return statment has
        # changed
		ctrs, _ = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			if len(ctr) >= 5: #fitellipse fn requires min 5 contours
				(x,y), (MA, ma), angle = cv2.fitEllipse(ctr)
				temparea = (math.pi)*MA*ma
				if(temparea>area):
					largestctr = ctr
					area = temparea
			

		ellipse = cv2.fitEllipse(largestctr)
		mask = np.zeros(img.shape, np.uint8)
		poly = cv2.ellipse(mask, ellipse, (255,255,255),-1)

		newimg = cv2.bitwise_and(thresh, thresh, mask=poly)
		xtest.append(newimg)

def preprocessTrainNumPixels():
	for i, img in enumerate(train_images):
		img = train_images[i].astype('uint8')

		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

		# contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
        # I think I'm using a diff openCV version where the return statment has
        # changed
		ctrs, _ = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		#Get contour containing largest number of pixels (area)
		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			temparea = cv2.contourArea(ctr)
			if temparea>area:
				largestctr = ctr
				area = temparea

		mask = np.zeros(img.shape, np.uint8)
		cv2.drawContours(mask, [largestctr], -1, 255, -1)

		newimg = cv2.bitwise_and(thresh, thresh, mask=mask)
		xtrain.append(newimg)

def preprocessTestNumPixels():
	for i, img in enumerate(test_images):
		img = test_images[i].astype('uint8')

		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

		# contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)
        # I think I'm using a diff openCV version where the return statment has
        # changed
		ctrs, _ = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		#Get contour containing largest number of pixels (area)
		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			temparea = cv2.contourArea(ctr)
			if temparea>area:
				largestctr = ctr
				area = temparea

		mask = np.zeros(img.shape, np.uint8)
		cv2.drawContours(mask, [largestctr], -1, 255, -1)

		newimg = cv2.bitwise_and(thresh, thresh, mask=mask)
		xtest.append(newimg)



xtrain = []
xtest = []

def save_processed(filename, dataset):
	with open (filename, 'wb') as f:
		pickle.dump(dataset, f)

""" Largest Rotated Rectangle """

print('Rectangle')
preprocessTrainMinRect()
preprocessTestMinRect()

save_processed("train_rect_1.pkl", xtrain)
save_processed("test_rect_1.pkl", xtest)

""" Largest Rotated Ellipse """

print('Ellipse')
preprocessTrainMinEllipse()
preprocessTestMinEllipse()

save_processed("train_ellipse_1.pkl", xtrain)
save_processed("test_ellipse_1.pkl", xtest)

""" Greatest Area Contour """

print('Countour')
preprocessTrainNumPixels()
preprocessTestNumPixels()

save_processed("train_pix_1.pkl", xtrain)
save_processed("test_pix_1.pkl", xtest)

#Example
plt.imshow(train_images[12])
plt.show()

plt.imshow(xtrain[12])
plt.show()

# plt.imshow(test_images[16])
# plt.show()

# plt.imshow(xtest[16])
# plt.show()


