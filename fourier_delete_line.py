import cv2 as cv
import numpy as np
import fr_ssszz as fr


if __name__ == "__main__":
    ## ตัวอย่างการใข้งาน ssszz lib ให้เต็มประสิทธิภาพ
    img=cv.imread("./input.png",cv.IMREAD_GRAYSCALE)
    deline = cv.imread("./kernel.png",cv.IMREAD_GRAYSCALE)

    deline_freq=fr.filterToFrequency(deline,s=img.shape)
    
    #รูปต้องทำเป็น mag ก่อนถึงจะพร้อมเอาไป x
    deline_mag=fr.filterfrequencyToMagnitude(deline_freq)
    
    imgfreq=fr.imgToFrequency(img)
    mag_img=fr.imgfrequencyToMagnitude(imgfreq)
    
    filtered_img = imgfreq*deline
    
    filtered_img = fr.invertFourierTransform(filtered_img)

    
    cv.imshow("magnitude_img",mag_img)
    cv.imshow("delineFilter_mag",deline_mag)
    cv.imshow("output",filtered_img)
    cv.imshow("original",img)
    cv.waitKey(0)
    cv.destroyAllWindows()