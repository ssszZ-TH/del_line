import math
import numpy as np
import cv2 as cv

img_h = 512
img_w = 512

gx, gy = np.meshgrid(np.linspace(-img_h/2, img_h/2, img_h), np.linspace(-img_w/2, img_w/2, img_w)) 

map_dst = np.sqrt(gx*gx+gy*gy)
map_dir = np.arctan2(gy,gx)
map_dir = (map_dir)*(180/np.pi)

# # Filter parameters ## default
# filt_rad1 = 50
# filt_rad2 = 150

# filt_ang1 = 30
# filt_ang2 = 150
# Filter parameters
filt_rad1 = 70
filt_rad2 = 150

filt_ang1 = 30
filt_ang2 = 150

filt = np.zeros([img_h,img_w,1], dtype=np.uint8)
filt1 = np.where(map_dst>=filt_rad1, 255, 0)
filt2 = np.where(map_dst<=filt_rad2, 255, 0)
filt3 = np.where(map_dir>=filt_ang1, 255, 0)
filt4 = np.where(map_dir<=filt_ang2, 255, 0)
filt = filt1 & filt2 & filt3 & filt4

cv.imwrite("output_filter.png", filt)