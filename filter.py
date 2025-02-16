# Python program to demonstrate erosion and  
# dilation of images. 
import cv2 
import numpy as np 
import argparse

ap = argparse.ArgumentParser(description="UTILS")
ap.add_argument("-i", "--input", required=True, type=str, default="no", help="required input image")
args = vars(ap.parse_args())
  
# Reading the input image 
img = cv2.imread(args["input"], 0) 
  
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((5,5), np.uint8) 
  
# The first parameter is the original image, 
# kernel is the matrix with which image is  
# convolved and third parameter is the number  
# of iterations, which will determine how much  
# you want to erode/dilate a given image.  
img_erosion = cv2.erode(img, kernel, iterations=1) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 
  
cv2.imshow('Input', img) 
cv2.imshow('Erosion', img_erosion) 
cv2.imshow('Dilation', img_dilation) 

cv2.imwrite("dilation.jpg", img_dilation)
  
cv2.waitKey(0) 