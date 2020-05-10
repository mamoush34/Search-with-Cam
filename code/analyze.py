import requests
import serverconfig
import scipy.misc
import cv2
import hyperparameters as hp


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
         


def predict(): 
    # iterations = 0
    # path = "../data/test-images.csv"
    # selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # for e,i in enumerate(os.listdir(path)):
    #     if i.startswith("4"):
    #         iterations += 1
    #         img = cv2.imread(os.path.join(path,i))
    #         selective_search.setBaseImage(img)
    #         selective_search.switchToSelectiveSearchFast()
    #         ssresults = selective_search.process()
    #         imout = img.copy()
    #         for e,result in enumerate(ssresults):
    #             if e < 2000:
    #                 x,y,w,h = result
    #                 timage = imout[y:y+h,x:x+w]
    #                 resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
    #                 img = np.expand_dims(resized, axis=0)
    #                 out= model.predict(img)
    #                 if out[0][0] > 0.65:
    #                     cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    #         plt.figure()
    #         plt.imshow(imout)
    #         break
    pass

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
    