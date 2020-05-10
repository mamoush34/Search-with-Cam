import os
import random
import numpy as np
from io import BytesIO
import urllib
import cv2
import ssl
from PIL import Image
from skimage import io
import tensorflow as tf
import hyperparameters as hp
import pandas as pd
import matplotlib.pyplot as plt
from boundingbox import Boundingbox
from HotEncoder import HotEncoder
from sklearn.model_selection import train_test_split
import threading
import queue
from keras.preprocessing.image import ImageDataGenerator




class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):

        if os.path.isfile("../data/train_images.npy") and os.path.isfile("../data/train_labels.npy"):
            print("Previous data file found. Loading npy files...")
            self.train_images = np.array(np.load("../data/train_images.npy"))
            self.train_labels = np.array(np.load("../data/train_labels.npy"))
        else:
            print("Making new data")
            #do not touch these 4 lines. Critical for multithreading.
            self.train_images = queue.Queue()
            self.train_labels = queue.Queue()     
            raw_images_count = pd.read_csv("../data/test-images.csv").shape[0]
            self.multithread_training_data(3000) 
            np.save("../data/train_images.npy", self.train_images)
            np.save("../data/train_labels.npy", self.train_labels)
       
        # # Mean and std for standardization
        # self.mean = np.zeros((3,))
        # self.std = np.ones((3,))
        # self.calc_mean_and_std(self.train_images)

        encoder = HotEncoder()
        Y_end = encoder.fit_transform(self.train_labels)

        #The splitting is done, so we can use it afterwards
        X_train, X_test, Y_train, Y_test = train_test_split(self.train_images, Y_end, test_size=0.10)

        self.train_X = X_train
        self.train_Y = Y_train
        self.test_X = X_test
        self.test_Y = Y_test

        self.train_data = self.augment_data(X_train, Y_train)
        self.test_data =  self.augment_data(X_test, Y_test)


    def multithread_training_data(self, num_images):
        print("Starting multithreading...")
        images_per_thread = num_images // hp.thread_count
        threads = []
        start = 0
        finish = images_per_thread
        for i in range(hp.thread_count - 1):
            thread = threading.Thread(target=self.create_training_data, args=(start, start + images_per_thread))
            thread.start() 
            threads.append(thread)
            start += images_per_thread
        remaining = num_images - start
        thread = threading.Thread(target=self.create_training_data, args=(start, start + remaining))
        thread.start()
        threads.append(thread)
        
        print(str(hp.thread_count) + " threads created.")
        print("Waiting for threads to finish...")
        for thread in threads:
            thread.join()
        print("All threads have finished. Unpacking results...")
        self.unpack_train_builder()

    def unpack_train_builder(self):
        train_images = self.train_images
        train_labels = self.train_labels
        arr_images = []
        arr_labels = []
        while not train_images.empty():
            images = train_images.get()
            labels = train_labels.get()
            assert len(labels) == len(images)
            if len(labels) != 0:
                for i in range(len(labels)):
                    arr_images.append(np.array(images[i]))
                    arr_labels.append(np.array(labels[i]))
        train_images = np.array(arr_images)
        train_labels = np.array(arr_labels)
        print(train_images.shape)
        print(train_labels.shape)
        self.train_images = train_images
        self.train_labels = train_labels

        
    
    
    def train_builder(function):
        def wrapper(self, *args):
            images, labels = function(self, *args)
            self.train_images.put(images)
            self.train_labels.put(labels)
        return wrapper

    @train_builder
    def create_training_data(self, start, finish):
        training_raw_images = pd.read_csv("../data/test-images.csv")
        class_names = pd.read_csv("../data/class-names.csv")
        training_annotations = pd.read_csv("../data/test-annotations.csv")
        
        train_images = []
        train_labels = []

        context = ssl._create_unverified_context()
        for i, row in training_raw_images.iloc[start: finish].iterrows():
            # print(str(i) + " image put into training set.")

            #All images are external images (URL Links). Code below downloads and analyzes them 
            image_name = os.path.splitext(row["image_name"])[0]
            raw_image_file = urllib.request.urlopen(row["image_url"], context=context)
            raw_image_PIL = Image.open(raw_image_file).convert("RGB")
            raw_image = np.asarray(raw_image_PIL) #just the image
            annotation = training_annotations[training_annotations["ImageID"] == image_name]  #annotation of the image          
            if annotation.empty:
                print("Detected empty annotation")
                continue
            label_name = annotation["LabelName"].values[0]
            label = class_names.loc[class_names["LabelName"] == label_name].squeeze()[1] #label for the image
            # print(label)
            img = cv2.resize(raw_image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)

            # # TO SEE THE RESCALED IMAGE, UNCOMMENT THIS
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            
            correct_bounding_boxes = []

            for i, row in annotation.iterrows():
                XMin,XMax,YMin,YMax = row["XMin"], row["XMax"], row["YMin"], row["YMax"]
                XMin *= img.shape[1]
                XMax *= img.shape[1]
                YMin *= img.shape[0]
                YMax *= img.shape[0]
                correct_bounding_boxes.append(Boundingbox(int(XMin), int(XMax), int(YMin), int(YMax)))
            
                # # TO SEE THE VISUALIZATION OF THE BOUNDING BOX, UNCOMMENT THIS
                # cv2.rectangle(img,(int(XMin),int(YMin)),(int(XMax),int(YMax)),(255,0,0), 2)
                # plt.figure()
                # plt.imshow(img)
                # plt.show()
            

            #now, segmentation using selective search
            cv2.setUseOptimized(True)
            selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            selective_search.setBaseImage(img)
            selective_search.switchToSelectiveSearchFast()
            boxes = selective_search.process()
            img_copy = img.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
        
            for i, box in enumerate(boxes):
                if i < hp.box_max_count and flag == 0:
                    for correct_bounding_box in correct_bounding_boxes:
                        x, y, w, h = box
                        bb = Boundingbox(x, x + w, y, y + h)
                        iou = self.calc_iof(correct_bounding_box, bb)
                        if counter < 30:
                            if iou > 0.70:
                                timage = img_copy[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (hp.img_size,hp.img_size), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(label)
                                counter += 1
                        else :
                            fflag =1
                        if falsecounter <30:
                            if iou < 0.3:
                                timage = img_copy[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (hp.img_size,hp.img_size), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append("Nothing")
                                falsecounter += 1
                        else :
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        flag = 1
        return train_images, train_labels

    
    
    def calc_iof(self, box1, box2):
        """Calculates the IOF between two bounding boxes. 
        The boxes passed into IOF must be from boundingbox class from boundingbox.py

        Arguments: box1 and box2, objects of boundingbox
        Returns IOF (intersection / union of box1 and box2)
        """
        x1 = max(box1.xmin, box2.xmin)
        x2 = min(box1.xmax, box2.xmax)
        y1 = max(box1.ymin, box2.ymin)
        y2 = min(box1.ymax, box2.ymax)
        if x2 < x1 or y2 < y1:
            return 0.0

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        i_area = dx * dy

        if i_area == 0: 
            return 0.0
        return i_area / float(box1.area + box2.area - i_area)


    def calc_mean_and_std(self, training_images):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        data_sample = np.copy(training_images)
        

        # Shuffle filepaths
        
        random.shuffle(data_sample)

        # Take sample of file paths
        data_sample = data_sample[:hp.preprocess_sample_size]
       

        #caculating pixelwise mean and std
        mean = np.zeros((3))
        std = np.zeros((3))
        for i in data_sample:
            for dim in range(3):
                image_channel = i[..., dim]
                mean[dim] += np.mean(image_channel)
                std[dim] += np.std(image_channel)
        mean /= len(data_sample)
        std /= len(data_sample)
        self.mean = mean
        self.std = std
        

        # ==========================================================

        print("Dataset mean: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0], self.mean[1], self.mean[2]))

        print("Dataset std: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0], self.std[1], self.std[2]))

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """
        #standardization
        for i in range(len(self.mean)):
            (img[...,i] - self.mean[i]) / self.std[i]
        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        img = img / 255.
        img = self.standardize(img)
        return img
    
    def augment_data(self, X_data, Y_data):
        augmenter = ImageDataGenerator(horizontal_flip= True, vertical_flip=True, rotation_range=90)
        return augmenter.flow(x=X_data, y=Y_data)
