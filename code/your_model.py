import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)
            
        arch = []
        arch.append(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu'))
        arch.append(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        arch.append(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
        arch.append(MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        arch.append(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

        arch.append(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))

        arch.append(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        arch.append(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        arch.append(Flatten())
        # 1st Fully Connected Layer
        arch.append(Dense(1024, input_shape=(224*224*3,), activation='relu'))
        # Add Dropout to prevent overfitting
        arch.append(Dropout(0.4))

        # 2nd Fully Connected Layer
        arch.append(Dense(512, activation='relu'))
        # Add Dropout
        arch.append(Dropout(0.4))


        # 2nd Fully Connected Layer
        arch.append(Dense(512, activation='relu'))
        # Add Dropout
        arch.append(Dropout(0.4))

        # Output Layer
        arch.append(Dense(15, activation='softmax'))

        self.architecture = arch

        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
