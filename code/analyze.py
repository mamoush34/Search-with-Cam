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
         


def predict(model, path): 
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)
    cv2.setUseOptimized(True)
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    boxes = selective_search.process()
    imout = image.copy()
    results = []
    ret_labels = []
    percentages = []
    train_labels = np.array(np.load("../data/train_labels.npy"))
    labels = sorted(train_labels)
    labels = list(dict.fromkeys(labels))
    for e,result in enumerate(boxes):
        if e < 2000:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (hp.img_size,hp.img_size), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            prob= model.predict(img)
            y_class = prob[0].argmax(axis=-1)
            if labels[y_class] != "Nothing" and prob[0][y_class] > 0.90:
                results.append(result)
                ret_labels.append(labels[y_class])
                percentages.append(int(prob[0][y_class]))
                #is there a way to know what label it is?
    ##add these if you're not running on gcp
    # plt.figure()
    # plt.imshow(imout)
    return normalize(results, hp.img_size, hp.img_size), list(ret_labels), list(percentages)

def segmentation(path):  
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)
    cv2.setUseOptimized(True)
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    boxes = selective_search.process()
    return normalize(boxes, hp.img_size, hp.img_size)
    