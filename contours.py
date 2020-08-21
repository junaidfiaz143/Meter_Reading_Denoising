import os
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser(description="UTILS")
ap.add_argument("-i", "--input", required=True, type=str, default="no", help="required input image")
args = vars(ap.parse_args())

input_image = args["input"]

output_folder = "output_" + input_image.split(".")[0]

if not os.path.exists(output_folder):
	os.makedirs(output_folder)
	print("[INFO] FOLDER CREATED!")
else:
	print("[ERROR] CREATING FOLDER!")

image = cv2.imread(input_image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (11, 11), cv2.BORDER_CONSTANT)

edged = cv2.Canny(blurred, 30, 150)

# Find contour and sort by contour area
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("No. of contours: ", len(contours))

disk = image.copy()
cv2.drawContours(disk, contours, -1, (0,255,0), 2)
cv2.imwrite(os.path.join(output_folder, "contours.png"), disk)
cv2.imshow("Contours", disk)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("Image", edged)

mask = np.zeros_like(image) 
cv2.drawContours(mask, contours, -1, (255,255,255), -1) 
cv2.imwrite(os.path.join(output_folder, "mask.png"), mask)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# threshold the gray image to binarize, and negate it
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

# create a mask for floodfill function, see documentation
h, w, clr = image.shape
mask = np.zeros((h+2, w+2), np.uint8)

# determine which contour belongs to a square or rectangle
for cnt in contours:
    poly = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
    if len(poly) == len(contours):
        # if the contour has 4 vertices then floodfill that contour with black color
        cnt = np.vstack(cnt).squeeze()
        _, binary, _, _ = cv2.floodFill(binary, mask, tuple(cnt[0]), 0)
# convert image back to original color
binary = cv2.bitwise_not(binary)        

cv2.imwrite(os.path.join(output_folder, "segment.png"), binary)
cv2.imshow('Segment', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()