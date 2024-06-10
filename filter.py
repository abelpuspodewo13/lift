from __future__ import print_function
from __future__ import division
import cv2 as cv
import argparse
import tkinter
SLIDER_MAX = 255
HUE_MAX = 179
title_window = 'Color Thresholding'
'''
global lower_value
global lower_saturation
global lower_hue
global upper_value
global upper_saturation
global upper_hue
'''

lower_hue = 0.0
lower_saturation = 0.0
lower_value = 0.0
upper_hue = 0.0
upper_saturation = 0.0
upper_value = 0.0

#赤Track Bar
def lower_hue_trackbar(val):
    global lower_hue 
    lower_hue = val
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    
#緑Track Bar
def lower_saturation_trackbar(val):
    global lower_saturation 
    lower_saturation = val
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (value, saturation, hue), (value, saturation, hue))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('saturation Filter', result)
    '''
    
#青Track Bar
def lower_value_trackbar(val):
    global lower_value 
    lower_value = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (value, saturation, hue), (value, saturation, hue))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('value Filter', result)
    '''
def upper_hue_trackbar(val):
    global upper_hue 
    upper_hue = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    
#緑Track Bar
def upper_saturation_trackbar(val):
    global upper_saturation
    upper_saturation = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (value, saturation, hue), (value, saturation, hue))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('saturation Filter', result)
    '''
    
#青Track Bar
def upper_value_trackbar(val):
    global upper_value 
    upper_value = val 
    filter_callback()
    #beta = ( 1.0 - alpha )
    #dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
    #dst = cv.addWeighted(src1, alpha, 0.0)
    '''
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, (value, saturation, hue), (value, saturation, hue))
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('value Filter', result)
    '''

def filter_callback():
    src = cv.imread(cv.samples.findFile(args.input))
    #cv.imshow('Awal', src)
    if src is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    hsv = cv.cvtColor(src,cv.COLOR_RGB2HSV)
    #cv.imshow('HSV', hsv)
    #tkinter.messagebox.showinfo(title=None, message=upper_hue, **options)
    mask = cv.inRange(hsv, (lower_hue, lower_saturation, lower_value), (upper_hue, upper_saturation, upper_value))
    #cv.imshow('mask', mask)
    result = cv.bitwise_and(src, src, mask=mask)
    cv.imshow('Filter', result)
    #cv.imwrite(cv.samples.findFile(args.input), result)
    

#Command Parser
parser = argparse.ArgumentParser(description='Code for Adding a Trackbar to our applications tutorial.')
parser.add_argument('--input', help='Path to the first input image.', default='./data/train/images/0180_1.jpg')
args = parser.parse_args()

cv.namedWindow(title_window)
lower_hue_name = 'Lower hue x %d' % HUE_MAX
lower_saturation_name = 'Lower saturation x %d' % SLIDER_MAX
lower_value_name = 'Lower value x %d' % SLIDER_MAX
upper_hue_name = 'Upper hue x %d' % HUE_MAX
upper_saturation_name = 'Upper saturation x %d' % SLIDER_MAX
upper_value_name = 'Upper value x %d' % SLIDER_MAX
cv.createTrackbar(lower_hue_name, title_window , 0, HUE_MAX, lower_hue_trackbar)
cv.createTrackbar(lower_saturation_name, title_window , 0, SLIDER_MAX, lower_saturation_trackbar)
cv.createTrackbar(lower_value_name, title_window , 0, SLIDER_MAX, lower_value_trackbar)
cv.createTrackbar(upper_hue_name, title_window , 0, HUE_MAX, upper_hue_trackbar)
cv.createTrackbar(upper_saturation_name, title_window , 0, SLIDER_MAX, upper_saturation_trackbar)
cv.createTrackbar(upper_value_name, title_window , 0, SLIDER_MAX, upper_value_trackbar)
# Show some stuff
#hue_trackbar(0)
#saturation_trackbar(0)
#value_trackbar(0)
filter_callback()


# Wait until user press some key
cv.waitKey(0) & 0xFF
