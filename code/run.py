import argparse
# from your_model import YourModel
# import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
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






os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# def parse_args():
#     """ Perform command-line argument parsing. """

#     parser = argparse.ArgumentParser(
#         description="Let's train some neural nets!")
#     parser.add_argument(
#         '--data',
#         default=os.getcwd() + '/../data/',
#         help='Location where the dataset is stored.')
#     parser.add_argument(
#         '--load-checkpoint',
#         default=None,
#         help='''Path to model checkpoint file (should end with the
#         extension .h5). Checkpoints are automatically saved when you
#         train your model. If you want to continue training from where
#         you left off, this is how you would load your weights. In
#         the case of task 2, passing a checkpoint path will disable
#         the loading of VGG weights.''')
#     parser.add_argument(
#         '--confusion',
#         action='store_true',
#         help='''Log a confusion matrix at the end of each
#         epoch (viewable in Tensorboard). This is turned off
#         by default as it takes a little bit of time to complete.''')
#     parser.add_argument(
#         '--evaluate',
#         action='store_true',
#         help='''Skips training and evaluates on the test set once.
#         You can use this to test an already trained model by loading
#         its checkpoint.''')

#     return parser.parse_args()

def make_prediction(model, image):
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
            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out= model.predict(img)
            if out[0][0] > 0.70:
                cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                results.append(result)
    # plt.figure()
    # plt.imshow(imout)
    return results


def main():
    """ Main function. """

    checkpoint_path = "./saved_models/"
    model_final = None
    #makes checkpoint folder if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    if os.path.isfile(checkpoint_path + "rcnn_vgg16_1.h5"):
        print("Found an existing model! Loading it...")
        model_final = tf.keras.models.load_model(checkpoint_path + "rcnn_vgg16_1.h5")
    else:
        datasets = Datasets(ARGS.data)

        vggmodel = VGG16(weights='imagenet', include_top=True)
        vggmodel.summary()   

        for layers in (vggmodel.layers)[:15]:
            print(layers)
            layers.trainable = False

        X= vggmodel.layers[-2].output
        predictions = Dense(20, activation="softmax")(X)
        model_final = Model(input = vggmodel.input, output = predictions)
        opt = Adam(lr=0.0001)

        model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
        model_final.summary()

        checkpoint = ModelCheckpoint(checkpoint_path + "rcnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

        hist = model_final.fit_generator(generator= datasets.train_data, steps_per_epoch= 10, epochs= 60, validation_data= datasets.test_data, validation_steps=2, callbacks=[checkpoint,early_stop])

    if os.path.isfile("../example_image/harbor.jpg"):
        image = plt.imread("../example_image/harbor.jpg")
        results = make_prediction(model_final, image)
        print(results)



# ARGS = parse_args()

main()
