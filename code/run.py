import os
import argparse
import tensorflow as tf
from your_model import YourModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np




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

    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "./your_model_checkpoints/"
    model.summary()

    #???
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    #makes checkpoint folder if it doesn't exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])


    checkpoint = ModelCheckpoint("rcnn_model", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

    hist = model.fit_generator(generator= datasets.train_data, steps_per_epoch= 10, epochs= 1000, validation_data= datasets.test_data, validation_steps=2, callbacks=[checkpoint,early_stop])


    #### I want to check in with this stuff, because I believe they do this to plot the bounding boxes, but they should've sued the 
    ### results found before.
    iterations = 0
    path = "../data/test-images.csv"
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    for e,i in enumerate(os.listdir(path)):
        if i.startswith("4"):
            iterations += 1
            img = cv2.imread(os.path.join(path,i))
            selective_search.setBaseImage(img)
            selective_search.switchToSelectiveSearchFast()
            ssresults = selective_search.process()
            imout = img.copy()
            for e,result in enumerate(ssresults):
                if e < 2000:
                    x,y,w,h = result
                    timage = imout[y:y+h,x:x+w]
                    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                    img = np.expand_dims(resized, axis=0)
                    out= model.predict(img)
                    if out[0][0] > 0.65:
                        cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow(imout)
            break

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
