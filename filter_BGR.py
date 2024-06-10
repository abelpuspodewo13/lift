from __future__ import print_function
from __future__ import division
import cv2 as cv
import argparse
import tkinter
import numpy as np
SLIDER_MAX = 255
red_MAX = 179
title_window = 'Color Thresholding'
'''
global lower_blue
global lower_green
global lower_red
global upper_blue
global upper_green
global upper_red
'''

lower_red = 0.0
lower_green = 0.0
lower_blue = 0.0
upper_red = 0.0
upper_green = 0.0
upper_blue = 0.0

#赤Track Bar
def lower_red_trackbar(val):
    global lower_red 
    lower_red = val
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    
#緑Track Bar
def lower_green_trackbar(val):
    global lower_green 
    lower_green = val
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (blue, green, red), (blue, green, red))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('green Filter', result)
    '''
    
#青Track Bar
def lower_blue_trackbar(val):
    global lower_blue 
    lower_blue = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (blue, green, red), (blue, green, red))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('blue Filter', result)
    '''
def upper_red_trackbar(val):
    global upper_red 
    upper_red = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    
#緑Track Bar
def upper_green_trackbar(val):
    global upper_green
    upper_green = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (blue, green, red), (blue, green, red))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('green Filter', result)
    '''
    
#青Track Bar
def upper_blue_trackbar(val):
    global upper_blue 
    upper_blue = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (blue, green, red), (blue, green, red))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('blue Filter', result)
    '''

def filter_callback():
    src = cv.imread(cv.samples.findFile(args.input))
    #cv.imshow('Awal', src)
    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    #cv.imshow('HSV', hsv)
    #tkinter.messagebox.showinfo(title=None, message=upper_red, **options)
    mask = cv.inRange(hsv, (lower_red, lower_green, lower_blue), (upper_red, upper_green, upper_blue))
    #cv.imshow('mask', mask)
    '''
    t=10
 
    thres = cv.inRange(src,(lower_blue-t,lower_green-t,lower_red-t),(upper_blue, upper_green, upper_red))
    #thres = cv.inRange(src,np.array([lower_blue-t,lower_green-t,lower_red-t]),np.array([upper_blue, upper_green, upper_red]))
    mask = 255- thres #inverse
    mask_bgr = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
    
    result = cv.max(src, mask_bgr)
    cv.imshow('Filter', result)
    #cv.imwrite(cv.samples.findFile(args.input), result)
    

#Command Parser
parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
parser.add_argument('--input', help='Path to the first input image.', default='./data/train/images/0180_1.jpg')
args = parser.parse_args()

cv.namedWindow(title_window)
lower_red_name = 'Lower red x %d' % red_MAX
lower_green_name = 'Lower green x %d' % SLIDER_MAX
lower_blue_name = 'Lower blue x %d' % SLIDER_MAX
upper_red_name = 'Upper red x %d' % red_MAX
upper_green_name = 'Upper green x %d' % SLIDER_MAX
upper_blue_name = 'Upper blue x %d' % SLIDER_MAX
cv.createTrackbar(lower_red_name, title_window , 0, SLIDER_MAX, lower_red_trackbar)
cv.createTrackbar(lower_green_name, title_window , 0, SLIDER_MAX, lower_green_trackbar)
cv.createTrackbar(lower_blue_name, title_window , 0, SLIDER_MAX, lower_blue_trackbar)
cv.createTrackbar(upper_red_name, title_window , 0, SLIDER_MAX, upper_red_trackbar)
cv.createTrackbar(upper_green_name, title_window , 0, SLIDER_MAX, upper_green_trackbar)
cv.createTrackbar(upper_blue_name, title_window , 0, SLIDER_MAX, upper_blue_trackbar)
# Show some stuff
#red_trackbar(0)
#green_trackbar(0)
#blue_trackbar(0)
filter_callback()


# Wait until user press some key
cv.waitKey(0) & 0xFF
