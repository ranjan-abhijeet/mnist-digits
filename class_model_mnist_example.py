import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

"""
3rd method of building deeplearning solution is using OOPS approach
in which we define a custom class inhereting from
tensorflow.keras.Model()
"""
class ModelClass(tf.keras.Model):


    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    
    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x 
    


def display_examples(examples, labels):
    """
    Plots images from the examples and labels passed.
    """
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    """
    Data returned by tf.keras.datasets.mnist.load_data() is in the following format:
        tuple of tuples: ((any, any), (any, any))
    """
    ((x_train, y_train), (x_test,y_test)) = tf.keras.datasets.mnist.load_data()
    if False:
        display_examples(x_test,y_test)

    """
    Converts the data from int8 to float32 so that normalization does
    not produce all 0s when divided by 255.
    Normalization i.e., dividing by 255 will translate the range to (0,1)
    which helps in faster convergence.
    """
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    """
    The input layer of the network expects the input data(in this code)
    to have dimensions (28,28,1), we are adding axis=-1 to expand the 
    dimensions from current (28,28). 
    """
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    """
    Compiling the model with optimizer, loss function, metric to target, etc.
    loss='categorical_crossentropy' expects input to be one-hot-encoded, 
    instead use loss='sparse_categorical_crossentropy'.
    """
    model = ModelClass()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    
    """
    Fitting the compiled model on training data, this is training set.
    Setting input data batch size, number of epochs, and validation_split.
    """
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    """
    Evaluating the model on data not seen by the trained model
    """
    model.evaluate(x_test, y_test, batch_size=64)