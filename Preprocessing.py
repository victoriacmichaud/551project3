import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import cv2

train_labels = pd.read_csv('./train_labels.csv')
train_images = pd.read_pickle('./train_images.pkl')
test_images = pd.read_pickle('test_images.pkl')

#print(train_images.shape)
#print(test_images.shape)

# plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
#plt.imshow(train_images[10])
#plt.show()

"""Preprocessing using OpenCV"""

def preprocessTrainMinRect():
	for i, img in enumerate(train_images):
		img = train_images[i].astype('uint8')

		#Threshold the image for values of 0 or 1
		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
		# plt.imshow(thresh)
		# plt.show()

		#Get contours of the binary image
		contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		#Get the area of the largest rectangle
		area = 0
		largestctr = ctrs[0]
		for ctr in ctrs:
			temparea = cv2.minAreaRect(ctr)[1][0]*cv2.minAreaRect(ctr)[1][1]
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

		newimg = cv2.bitwise_and(img, mask) #black and white image containing only the largest number
		xtrain.append(newimg)

def preprocessTestMinRect():
	for i, img in enumerate(test_images):
		img = test_images[i].astype('uint8')

		#Threshold the image for values of 0 or 1
		ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

		#Get contours of the binary image
		contoured, ctrs, heir = cv2.findContours(thresh,1,cv2.CHAIN_APPROX_SIMPLE)

		#Get the area of the largest rectangle and contour corresponding to it
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
		cv2.fillPoly(mask,poly,255) #creating a mask over the irrelevant number contours, replace with black

		newimg = cv2.bitwise_and(img, mask) #black and white image containing only the largest number
		xtest.append(newimg)


xtrain = []
xtest = []
preprocessTrainMinRect()
preprocessTestMinRect()

#Example
# plt.imshow(xtrain[16])
# plt.show()

def save_processed(filename, dataset):
	with open (filename, 'wb') as f:
		pickle.dump(dataset, f)

#save_processed("train_rect_1.pkl", xtrain)
#save_processed("test_rect_1.pkl", xtest)


