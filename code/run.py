import os, keras
import argparse
import tensorflow as tf
from your_model import YourModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense
from keras import Model
from keras.applications.vgg16 import VGG16





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
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(datasets)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    datasets = Datasets(ARGS.data)

    vggmodel = VGG16(weights='imagenet', include_top=True)
    vggmodel.summary()   

    for layers in (vggmodel.layers)[:15]:
        print(layers)
        layers.trainable = False

    X= vggmodel.layers[-2].output
    predictions = Dense(2, activation="softmax")(X)
    model_final = Model(input = vggmodel.input, output = predictions)
    opt = Adam(lr=0.0001)

    # model = YourModel()
    # model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "./your_model_checkpoints/"
    # model.summary()

    #???
    if ARGS.load_checkpoint is not None:
        model_final.load_weights(ARGS.load_checkpoint)

    #makes checkpoint folder if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
    model_final.summary()

    # # Compile model graph
    # model.compile(
    #     optimizer=model.optimizer,
    #     loss=model.loss_fn,
    #     metrics=["accuracy"])

    
    # checkpoint = ModelCheckpoint("rcnn_model", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', save_freq=1)
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')

    checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')


    trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    traindata = trdata.flow(x=datasets.train_X, y=datasets.train_Y)
    tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    testdata = tsdata.flow(x=datasets.test_X, y=datasets.test_Y)

    # print(f"Train data X shape: {traindata.x.shape}")
    # print(f"Test data X shape: {testdata.x.shape}")
    # print(f"Train data Y shape: {traindata.y.shape}")
    # print(f"Test data Y shape: {testdata.y.shape}")



    hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 10, epochs= 1000, validation_data= testdata, validation_steps=2, callbacks=[checkpoint,early])

    #### I want to check in with this stuff, because I believe they do this to plot the bounding boxes, but they should've sued the 
    ### results found before.
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

    # model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    # checkpoint_path = "./your_model_checkpoints/"
    # model.summary()

    # if ARGS.load_checkpoint is not None:
    #     model.load_weights(ARGS.load_checkpoint)

    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)

    # # Compile model graph
    # model.compile(
    #     optimizer=model.optimizer,
    #     loss=model.loss_fn,
    #     metrics=["sparse_categorical_accuracy"])

    # if ARGS.evaluate:
    #     test(model, datasets.test_data)
    # else:
    #     train(model, datasets, checkpoint_path)

# Make arguments global
ARGS = parse_args()

main()
