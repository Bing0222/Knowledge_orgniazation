# Using pix2pix makes src
"""
1. using two veido and reading frames get iamges
2. images cat them
3. output dataset

"""

import cv2
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(
    description='This is a script to make a dataset for pix2pix with 2 videos.'
)
parser.add_argument('--v1', required=True, help='Path to video1, generate the 1st image.')
parser.add_argument('--v2', required=True, help='Path to video2, generate the 2nd image.')
parser.add_argument('--outpath', default='frames2image', help='path to save the images.')
parser.add_argument('--prefix', default='', help='prefix of image name.')
parser.add_argument('--count', type=int, default=-1, help='the number of images to gen.')
parser.add_argument('--width', type=int, default=450, help='the width of each frame to resize.')
parser.add_argument('--height', type=int, default=420, help='the height of each frame to resize.')

args = parser.parse_args()

outpath = args.outpath
count = args.count
target_w = args.width
target_h = args.height

cap1 = cv2.VideoCapture(args.v1)
cap2 = cv2.VideoCapture(args.v2)

# get size 
w1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
h1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
w2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
h2 = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)

if w1 != w2 or h1 != h2:
    print('The shapes of the 2 videos are different!')
    print('But we can continue...')

# get fps
frameNum1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frameNum2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

max_count = frameNum1 if frameNum1 < frameNum2 else frameNum2
image_count = max_count if count < 0 or count > max_count else count

start_time = time.time()

for i in range(image_count):
    _,fram1 = cap1.read()
    _,fram2 = cap2.read()
    # resize 
    fram1 = cv2.resize(fram1,(target_w,target_h),interpolation=cv2.INTER_CUBIC)
    fram2 = cv2.resize(fram2,(target_w,target_h),interpolation=cv2.INTER_CUBIC)

    img = np.concatenate([fram1,fram2],axis=1)
    if i > 0 and i % 20 == 0:
        end_time = time.time()
        print('{}/{} generation pictures, time{:.2f}s'.format(i,image_count,end_time-start_time))
end_time = time.time()
print("\n Generation{} pictures, Using{:.2f}s".format(image_count,end_time-start_time))


cap1.release()
cap2.release()