import argparse
from preprocess import Datasets
import os, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from HotEncoder import HotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
import hyperparameters as hp


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.''')

    return parser.parse_args()

def make_prediction(model, image):
    """ 
    Finds the coordinates of the bounding boxes that satisfy a threshold.
    The coordinates get appended in the order of xMin, xMax, yMin, YMax to a 1D list.
    Inputs
    - image: the image to detect objects for
    - model: the trained model that will do the predictions
    Returns
    - results: list of coordinates of the bounding boxes that satisfy the threshold.
    """
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
                results.append(x)
                results.append(x + w)
                results.append(y)
                result.append(y+ h)
    return results


def main():
    """ Main function. """

    checkpoint_path = "./saved_models/"
    model_final = None
    #makes checkpoint folder if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    #Loading model if user requested it
    if ARGS.load_checkpoint is not None:
        if ARGS.load_checkpoint.endswith('.h5') and os.path.isfile(ARGS.load_checkpoint):
                print("Found an existing model! Loading it...")
                model_final = tf.keras.models.load_model(ARGS.load_checkpoint)
                model_final.summary()
        else:
            print("Error: Pass in h5 file of the model!!")
            return 
    else:
        ### Load the data
        datasets = Datasets(ARGS.data)

        vggmodel = VGG16(weights='imagenet', include_top=True)
        vggmodel.summary()   

        ### Freezes every layeer in vggmodel
        for layers in (vggmodel.layers)[:15]:
            print(layers)
            layers.trainable = False

        X= vggmodel.layers[-2].output

        #A connected layer is added for predictions
        predictions = Dense(hp.num_classes, activation="softmax")(X)
        model_final = Model(input = vggmodel.input, output = predictions)
        opt = Adam(lr= hp.learning_rate)

        model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
        model_final.summary()
        
        #Training configurations are set and training is performed through fit_generator.
        checkpoint = ModelCheckpoint(checkpoint_path + "rcnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

        model_final.fit_generator(generator= datasets.train_data, steps_per_epoch= 10, epochs= 1000, validation_data= datasets.test_data, validation_steps=2, callbacks=[checkpoint,early_stop])


ARGS = parse_args()

main()
