import os
import random
import numpy as np
from io import BytesIO
import urllib
from cv2 import cv2
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



class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        train_images, train_labels = self.create_training_data()
        print("READ")
        self.data_path = data_path

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.category_num

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std(train_images)

        encoder = HotEncoder()
        Y_end = encoder.fit_transform(train_labels)

        #The splitting is done, so we can use it afterwards
        X_train, X_test, Y_train, Y_test = train_test_split(train_images, Y_end, test_size=0.10)

        self.train_data = self.augment_data(X_train, Y_train)
        self.test_data =  self.augment_data(X_test, Y_test)


        # # Setup data generators
        # self.train_data = self.get_data(
        #     os.path.join(self.data_path, "train/"), True, True)
        # self.test_data = self.get_data(
        #     os.path.join(self.data_path, "test/"), False, False)

    """Calculates the IOF between two bounding boxes. 
    The boxes passed into IOF must be from boundingbox class from boundingbox.py

    Arguments: box1 and box2, objects of boundingbox
    Returns IOF (intersection / union of box1 and box2)
    """
    def create_training_data(self):
        training_raw_images = pd.read_csv("../data/test-images.csv")
        class_names = pd.read_csv("../data/class-names.csv")
        training_annotations = pd.read_csv("../data/test-annotations.csv")
        
        train_images = []
        train_labels = []



        context = ssl._create_unverified_context()
        for i, row in training_raw_images.iterrows():

            #All images are external images (URL Links). Code below downloads and analyzes them 
            image_name = os.path.splitext(row["image_name"])[0]
            raw_image_file = urllib.request.urlopen(row["image_url"], context=context)
            raw_image_PIL = Image.open(raw_image_file).convert("RGB")
            raw_image = np.asarray(raw_image_PIL) #just the image
            
            annotation = training_annotations[training_annotations["ImageID"] == image_name]  #annotation of the image          
            
            label_name = annotation["LabelName"].values[0]
            label = class_names.loc[class_names["LabelName"] == label_name].squeeze()[1] #label for the image
            
            img = cv2.resize(raw_image, dsize=(hp.img_size, hp.img_size), interpolation=cv2.INTER_AREA)

            #TO SEE THE RESCALED IMAGE, UNCOMMENT THIS
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            # break

            #calculates bounding box, based on the image scale
            XMin,XMax,YMin,YMax = annotation["XMin"], annotation["XMax"], annotation["YMin"], annotation["YMax"]
            XMin *= img.shape[1]
            XMax *= img.shape[1]
            YMin *= img.shape[0]
            YMax *= img.shape[0]

            
            correct_bounding_box = Boundingbox(int(XMin), int(XMax), int(YMin), int(YMax))

            #TO SEE THE VISUALIZATION OF THE BOUNDING BOX, UNCOMMENT THIS
            # cv2.rectangle(img,(int(XMin),int(YMin)),(int(XMax),int(YMax)),(255,0,0), 2)
            # plt.figure()
            # plt.imshow(img)
            # plt.show()
            # break
            

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
                print(i)
                if i < hp.box_max_count and flag == 0:
                    x, y, w, h = box
                    bb = Boundingbox(x, x + w, y, y + h)
                    iou = self.calc_iof(correct_bounding_box, bb)
                    if counter < 30:
                        print(iou)
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
            return np.array(train_images), np.array(train_labels)

    
             
    def calc_iof(self, box1, box2):
        print(box1.xmin)
        dx = abs(max(box1.xmin, box2.xmin) - min(box1.xmax, box2.xmax))
        dy = abs(max(box1.ymin, box2.ymin) - min(box1.ymax, box2.ymax))
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
        data_sample = training_images
        

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
        augmenter = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip= True, vertical_flip=True, rotation_range=90)
        return augmenter.flow(x=X_data, y=Y_data)

    # def get_data(self, path, shuffle, augment):
    #     """ Returns an image data generator which can be iterated
    #     through for images and corresponding class labels.

    #     Arguments:
    #         path - Filepath of the data being imported, such as
    #                "../data/train" or "../data/test"
    #         shuffle - Boolean value indicating whether the data should
    #                   be randomly shuffled.
    #         augment - Boolean value indicating whether the data should
    #                   be augmented or not.

    #     Returns:
    #         An iterable image-batch generator
    #     """

    #     if augment:
    #         #augmentation
    #         data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #             preprocessing_function=self.preprocess_fn, 
    #             shear_range=0.2,
    #             zoom_range=0.2,
    #             horizontal_flip=True,
    #             fill_mode='nearest'
    #             )

    #         # ============================================================
    #     else:
    #         # Don't modify this
    #         data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #             preprocessing_function=self.preprocess_fn)

    #     #setting the image size
    #     img_size = hp.img_size

    #     classes_for_flow = None

    #     # Make sure all data generators are aligned in label indices
    #     if bool(self.idx_to_class):
    #         classes_for_flow = self.classes

    #     # Form image data generator from directory structure
    #     data_gen = data_gen.flow_from_directory(
    #         path,
    #         target_size=(img_size, img_size),
    #         class_mode='sparse',
    #         batch_size=hp.batch_size,
    #         shuffle=shuffle,
    #         classes=classes_for_flow)

    #     # Setup the dictionaries if not already done
    #     if not bool(self.idx_to_class):
    #         unordered_classes = []
    #         for dir_name in os.listdir(path):
    #             if os.path.isdir(os.path.join(path, dir_name)):
    #                 unordered_classes.append(dir_name)

    #         for img_class in unordered_classes:
    #             self.idx_to_class[data_gen.class_indices[img_class]] = img_class
    #             self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
    #             self.classes[int(data_gen.class_indices[img_class])] = img_class

    #     return data_gen