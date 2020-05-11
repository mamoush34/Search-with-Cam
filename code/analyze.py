import requests
import serverconfig
import scipy.misc
import cv2
import hyperparameters as hp
import numpy as np



def normalize(boxes, imgwidth, imgheight):
    """
    Normalize function takes in a list of bounding boxes, and normalizes
    each coordinates based on imgwidth and imgheight

    INPUT: boxes - number of bounding boxes, each with x, y, w, h
           imgwidth - width of the image
           imgheight - height of the image
    OUTPUT: normalized points
    """
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
    """ 
    Finds the coordinates of the bounding boxes that satisfy a threshold.
    The coordinates get appended in the order of xMin, xMax, yMin, YMax to a 1D list.
    Inputs
    - image: the image to detect objects for
    - model: the trained model that will do the predictions
    Returns
    - results: list of coordinates of the bounding boxes that satisfy the threshold.
    """
    #gets the image
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)
    #segmentation
    cv2.setUseOptimized(True)
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    boxes = selective_search.process()
    imout = image.copy()
    #initialize arrays and load train_labels for label matching
    results = []
    ret_labels = []
    percentages = []
    train_labels = np.array(np.load("../data/train_labels.npy"))
    labels = sorted(train_labels)
    labels = list(dict.fromkeys(labels))
    #loop through all the boxes
    for e,result in enumerate(boxes):
        if e < 2000:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (hp.img_size,hp.img_size), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            prob= model.predict(img)
            y_class = prob[0].argmax(axis=-1)
            #only cases where class is nothing or the given class has higher than 0.85 accuracy
            if labels[y_class] != "Nothing" and prob[0][y_class] > 0.85:
                results.append(result)
                ret_labels.append(labels[y_class])
                percentages.append(int(prob[0][y_class]))
    return normalize(results, hp.img_size, hp.img_size), list(ret_labels), list(percentages)

def segmentation(path):  
    """
    Segments the image, given the path, and returns the normalized bounding boxes.
    INPUT: path - image path
    OUTPUT: normalized segmented points
    """
    #load the image
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)
    cv2.setUseOptimized(True)
    #segmentation
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(image)
    selective_search.switchToSelectiveSearchFast()
    #get boxes and noramlize them
    boxes = selective_search.process()
    return normalize(boxes, hp.img_size, hp.img_size)
    