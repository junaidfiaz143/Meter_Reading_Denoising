import cv2
import numpy as np
import pytesseract
from PIL import Image
import argparse

ap = argparse.ArgumentParser(description="UTILS")
ap.add_argument("-i", "--input", required=True, type=str, default="no", help="required input image")
args = vars(ap.parse_args())

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract"

output_path = ""

def get_String(img_path):
    #read image
    img = cv2.imread(img_path)
    #convert it to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #apply some dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    #save image
    cv2.imwrite(output_path + "img2.png", img)
    
    #apply threshold to get image only with black&white color
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #save image
    cv2.imwrite(output_path + "img3.png", img)
    
    #read text from images
    result = pytesseract.image_to_string(Image.open(output_path + "img3.png"))
    #return text results
    return result

print(get_String(output_path + args["input"]))