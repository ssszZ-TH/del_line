import math
import numpy as np
import cv2 as cv


if __name__ == "__main__":

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
    filt_rad1 = 50
    filt_rad2 = 150

    filt_ang1 = 40
    filt_ang2 = 130

    filt = np.zeros([img_h,img_w,1], dtype=np.uint8)
    filt1 = np.where(map_dst>=filt_rad1, 255, 0)
    filt2 = np.where(map_dst<=filt_rad2, 255, 0)
    filt3 = np.where(map_dir>=filt_ang1, 255, 0)
    filt4 = np.where(map_dir<=filt_ang2, 255, 0)
    filt = filt1 & filt2 & filt3 & filt4

    #convert type to uint8=spa 
    filt = filt.astype(np.uint8)
    filt1 = filt1.astype(np.uint8)
    filt2 = filt2.astype(np.uint8)
    filt3 = filt3.astype(np.uint8)
    filt4 = filt4.astype(np.uint8)

    filt_flip = cv.flip(filt,0)
    filt_flip = cv.flip(filt_flip,1)
    
    filtmain = filt+filt_flip
    
    cv.imwrite("kernel.png", filtmain)
    
    cv.imshow("filt_step1",filt)
    cv.imshow("filt_step2",filt_flip)
    cv.imshow("filt_step3",filtmain)
    
    cv.waitKey(0)
    cv.destroyAllWindows()