import cv2 as cv
import numpy as np
import fr_ssszz as fr


if __name__ == "__main__":
    ## ตัวอย่างการใข้งาน ssszz lib ให้เต็มประสิทธิภาพ
    img=cv.imread("./input.png",cv.IMREAD_GRAYSCALE)
    deline = cv.imread("./kernel_del_x_line.png",cv.IMREAD_GRAYSCALE)

    # dsize
    dsize = (img.shape[1],img.shape[0])

    # resize image
    deline_filt = cv.resize(deline, dsize, interpolation=cv.INTER_AREA)
    deline_filt = np.uint8(deline_filt)
    
    imgfreq=fr.imgToFrequency(img)
    mag_img=fr.imgfrequencyToMagnitude(imgfreq)
    
    deline_filt//=200
    filtered_img = imgfreq*deline_filt

    
    filtered_img = fr.invertFourierTransform(filtered_img)
    filtered_img_norm = cv.normalize(filtered_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


    cv.imwrite("output.png",filtered_img_norm)
    
    cv.imshow("magnitude_img",mag_img)
    cv.imshow("output",filtered_img)
    cv.imshow("output_normalize",filtered_img_norm)
    cv.imshow("original",img)
    cv.waitKey(0)
    cv.destroyAllWindows()