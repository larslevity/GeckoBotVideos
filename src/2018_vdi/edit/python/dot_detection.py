# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 12:15:01 2018

@author: AmP

from:
https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection

https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
"""

import cv2
import numpy as np
from skimage import measure
from imutils import contours
import imutils


def get_dot_coords(filename, debug=False):
    x, y = [], []

    def debug_helper(img, name='img'):
        if debug:
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # Load RGB image
    img = cv2.imread(filename,1)
    img = cv2.medianBlur(img,5)

    # detect green:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([65, 60, 60])
    upper_green = np.array([80, 255, 255])
    thresh = cv2.inRange(img_hsv, lower_green, upper_green)

    debug_helper(thresh, 'threshold')


    # filter noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    debug_helper(thresh, 'filter out noise')

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
     
    # loop over the unique components
    for label in np.unique(labels):
    	# if this is the background label, ignore it
    	if label == -1:
    		continue
     
    	# otherwise, construct the label mask and count the
    	# number of pixels 
    	labelMask = np.zeros(thresh.shape, dtype="uint8")
    	labelMask[labels == label] = 255
    	numPixels = cv2.countNonZero(labelMask)
     
    	# if the number of pixels in the component is sufficiently
    	# large, then add it to our mask of "large blobs"
    	print numPixels
    	debug_helper(labelMask, 'mask of label {}'.format(label))
    	if numPixels > 80:
    	    mask = cv2.add(mask, labelMask)



    debug_helper(mask, 'mask of Large Bobbles')



    # find the contours in the mask, then sort them from left to
    # right
    try:
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
    except ValueError:
        return x, y

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (xR, yR, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        x.append(int(cX))
        y.append(int(cY))
        cv2.circle(img, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), 3)
        cv2.putText(img, "{}".format(i + 1), (xR, yR - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    debug_helper(img, 'found bright spots')

#    # Look for circles in it
#    HOUGH_GRADIENT = 3  # cv2.HOUGH_GRADIENT = 3
#    circles = cv2.HoughCircles(mask,HOUGH_GRADIENT,2,20,
#                                param1=50,param2=12,
#                                minRadius=1,maxRadius=10)
#
#
#    if circles is None:
#        return [], []
#    circles = np.uint16(np.around(circles))
#    x = []
#    y = []
#    
#    for i in circles[0,:]:
#        x.append(i[0])
#        y.append(i[1])
    
    
    return x, y



if __name__ == "__main__":
    filename = 'vlcsnap-00001.png'
    
    print get_dot_coords(filename, debug=True)
