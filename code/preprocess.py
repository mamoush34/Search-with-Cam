import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):
        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.category_num

        # Mean and std for standardization
        self.mean = np.zeros((3,))
        self.std = np.ones((3,))
        self.calc_mean_and_std()

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

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

    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:
            #augmentation
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn, 
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
                )

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        #setting the image size
        img_size = hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen
