import numpy as np
from sklearn.preprocessing import LabelBinarizer

class HotEncoder(LabelBinarizer):

    def transform(self, y):
        """
        The method that hot encodes categorical data into binary representation.
        """
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1- Y))
        else :
            return Y

    def inverse_transform(self, Y, threshold=None):
        """
        The method that once given hot encoded labels due to transform, reverts back to original.
        """
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
