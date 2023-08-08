import cv2
 
src = cv2.imread('./kernel.png', cv2.IMREAD_UNCHANGED)

# dsize
dsize = (1920,1080)

# resize image
output = cv2.resize(src, dsize, interpolation = cv2.INTER_AREA)

cv2.imwrite('kernel_new.png',output) 