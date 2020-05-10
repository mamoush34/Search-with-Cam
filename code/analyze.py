import requests
import serverconfig
import scipy.misc
import cv2
import hyperparameters as hp
import numpy as np


def normalize(boxes, imgwidth, imgheight):
    points = []
    for i, box in enumerate(boxes):
         x, y, w, h = box
         xmin, xmax, ymin, ymax = x, x+w, y, y+h
         points.append(xmin / imgwidth)
         points.append(xmax / imgwidth)
         points.append(ymin / imgheight)
         points.append(ymax / imgheight)
    return points
         


def predict(model, image): 
    cv2.setUseOptimized(True)
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    boxes = selective_search.process()
    imout = image.copy()
    results = []
    for e,result in enumerate(boxes):
        if e < 2000:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (hp.img_size,hp.img_size), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out= model.predict(img)
            if out[0][0] > 0.70:
                ##add these if you're not running on gcp
                # cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                results.append(x)
                results.append(x + w)
                results.append(y)
                result.append(y+ h)
    ##add these if you're not running on gcp
    # plt.figure()
    # plt.imshow(imout)
    return results

def segmentation(path):  
    print(path)    
    image = cv2.imread("/Users/andrewkim/Desktop/Search-with-Cam/communication/rawimage/cars.jpg")
    image = cv2.resize(image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)
    cv2.setUseOptimized(True)
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    boxes = selective_search.process()
    return normalize(boxes, hp.img_size, hp.img_size)
    