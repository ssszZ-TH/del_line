import numpy as np
import cv2 as cv

## นี่เป็น library เกี่ยวกับ fourier ที่ผมสปาสร้างขึ้นมาใช้เอง
## หลักการเป็นเเบบ functional language เเต่พอถูก import
## ไปใช้ มันจะกลายเป็น OOP ทันที 

def getSobelX():
    return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
def getSobelY():
    return np.array([[-1, -2, -1],
                        [0, 0, 0], 
                        [1, 2, 1]])
    
def filterToFrequency(filter,s):
    # Fourier Transform of Filter Sobel
    # เเปลง filter เป็น frequency domain
    sobel_freq = np.fft.fft2(filter, s=s)
    sobel_freq_shifted = np.fft.fftshift(sobel_freq)
    return sobel_freq_shifted
    
def imgToFrequency(image):
    # เเปลง รุป เป็น frequency domain
    image_freq = np.fft.fft2(image)
    #shift frequency domain เป็น
    image_freq_shifted = np.fft.fftshift(image_freq)
    return image_freq_shifted

def imgfrequencyToMagnitude(image):
    image_real = np.real(image)
    img_imagine = np.imag(image)
    image_magnitude = np.sqrt(image_real**2 + img_imagine**2)
    image_magnitude = np.log(1+image_magnitude)
    image_magnitude = cv.normalize(image_magnitude,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
    return image_magnitude

def filterfrequencyToMagnitude(image):
    filter_magnitude = np.abs(image)
    return filter_magnitude

def invertFourierTransform(fr):
    image = np.fft.ifftshift(fr)
    image = np.fft.ifft2(image)
    image = np.abs(image)
    image = np.uint8(image)
    return image

if __name__=="__main__":
    ## ตัวอย่างการใข้งาน ssszz lib ให้เต็มประสิทธิภาพ
    img=cv.imread("./input.png",cv.IMREAD_GRAYSCALE);
    
    sobelx=getSobelX()
    sobely=getSobelY()
    
    sobelxfreq=filterToFrequency(sobelx,s=img.shape)
    sobelyfreq=filterToFrequency(sobely,s=img.shape)
    
    #รูปต้องทำเป็น mag ก่อนถึงจะพร้อมเอาไป x
    mag_sobelx=filterfrequencyToMagnitude(sobelxfreq)
    mag_sobely=filterfrequencyToMagnitude(sobelyfreq)
    
    imgfreq=imgToFrequency(img)
    mag_img=imgfrequencyToMagnitude(imgfreq)
    
    filtered_img_x = imgfreq*mag_sobelx
    filtered_img_y = imgfreq*mag_sobely
    
    filtered_img_x = invertFourierTransform(filtered_img_x)
    filtered_img_y = invertFourierTransform(filtered_img_y)
    
    cv.imshow("magnitude_img",mag_img)
    cv.imshow("sobelx_freq",mag_sobelx)
    cv.imshow("sobely_freq",mag_sobely)
    cv.imshow("filter image x",filtered_img_x)
    cv.imshow("filter image y",filtered_img_y)
    cv.imshow("original",img)
    cv.waitKey(0)
    cv.destroyAllWindows()