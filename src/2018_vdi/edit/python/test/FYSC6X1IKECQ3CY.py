# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt


filename = 'vlcsnap-0000{}.png'
center = []

debug = True


def debug_helper(img, name='img'):
        if debug:
            cv2.imshow(name, img)
            cv2.waitKey(0)



for i in range(1, 8):
    # for i in [3]:

    img = cv2.imread(filename.format(i), 1)
    img = cv2.medianBlur(img, 5)

    # detect green:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 60, 60])
    upper_red = np.array([10, 255, 255])
    gray = cv2.inRange(img_hsv, lower_red, upper_red)

#    cv2.imshow('gray', gray)
#    cv2.waitKey(0)

    # Adaptive Guassian Threshold is to detect sharp edges in the Image.
    # For more information Google it.
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 3.5)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=1)

#    cv2.imshow('gray', gray)
#    cv2.waitKey(0)

    # detect circles in the image
#    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=11, minDist=101,  # Works for Picture 1
#                               param1=1,
#                               param2=100, minRadius=0, maxRadius=20)    

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=13, minDist=101,  # works for 2
                               param1=1,
                               param2=100, minRadius=0, maxRadius=20)

    # ensure at least some circles were found
    if circles is not None:
        # sort
        idx_sorted = {i: 0 for i in range(4)}
        if len(list(circles[0, :, 0])) == 4:
            x = list(circles[0, :, 0])
            x_sorted = sorted(x)
            y = list(circles[0, :, 1])
            y_sorted = sorted(y)

            l12 = [y.index(y_sorted[0]), y.index(y_sorted[1])]
            l34 = [y.index(y_sorted[2])]
            y[l34[0]] = np.nan
            l34.append(y.index(y_sorted[3]))
            x12 = list(x)
            x34 = list(x)
            x12[l34[0]] = np.nan
            x12[l34[1]] = np.nan

            idx_sorted[x12.index(min([x[i] for i in l12]))] = 0
            idx_sorted[x12.index(max([x[i] for i in l12]))] = 1
            x34[l12[0]] = np.nan
            x34[l12[1]] = np.nan
            idx_sorted[x34.index(min([x[i] for i in l34]))] = 2
            idx_sorted[x34.index(max([x[i] for i in l34]))] = 3

            centers = [None]*4
        else:
            centers = 'Not 4 circles detected'
        # convert the (x, y) coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for idx, (x, y, r) in enumerate(circles):
            # draw the circle in the output image,
            # then draw a rectangle in the image
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x-5, y-5), (x+5, y+5), (0, 128, 255), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = .8
            col = (255, 0, 0)

            if type(centers) == list:
                # print x, y, idx, idx_sorted[idx]
                cv2.putText(img, str(idx_sorted[idx]), (x, y-10), font,
                            fontscale, col, 2)
                centers[idx_sorted[idx]] = (x, y)
            debug_helper(img)

    # Display the resulting frame
    print centers
#    cv2.imshow('gray', gray)
#    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    center.append(centers)


c_y = [[-i[j][1] for i in center] for j in range(4)]
c_x = [[-i[j][0] for i in center] for j in range(4)]


delta = [0, 20, 40, 50]

for i in range(4):
    plt.plot(c_y[i], label='y{}'.format(i))
plt.legend()

plt.figure()
for i in range(4):
    plt.plot(c_x[i], label='x{}'.format(i))
plt.legend()
