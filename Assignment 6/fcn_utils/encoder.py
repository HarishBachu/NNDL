import numpy as np 
import tensorflow as tf 

from tensorflow.keras import models, layers 

class ConvBlock(layers.Layer):
    def __init__(self, nconvs, nKernels, kernelSize, padding="same", strides=1, activation="relu"):
        super(ConvBlock, self).__init__() 

        self.conv = models.Sequential([
            layers.Conv2D(
                filters=nKernels, 
                kernel_size=kernelSize, 
                padding=padding, strides=strides, 
                activation=activation
            )
        ]) * nconvs 

        self.pool = layers.MaxPool2D((2, 2), strides=(2, 2)) 

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        pooled = self.pool(x) 
        return input_tensor, pooled 

class ConvDropout(layers.Layer):
    def __init__(self):
        super(ConvDropout, self).__init__() 
        self.conv = layers.Conv2D(4096, (7, 7), activation="relu", padding="same")
        self.dropout = layers.Dropout(0.5) 

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.dropout(x)
        return x 

class ConvCrop(layers.Layer):
    def __init__(self, n_classes, kernelSize, strides):
        super(ConvCrop, self).__init__() 
        self.deConv = layers.Conv2DTranspose(n_classes, kernel_size=kernelSize, strides=strides)
        self.crop = layers.Cropping2D(((1, 1), (1, 1))) 

    def call(self, input_tensor):
        x = self.deConv(input_tensor)
        x = self.crop(x)
        return x 

         

class Encoder(layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__() 
        
        self.conv1 = ConvBlock(2, 64, (3, 3))
        self.conv2 = ConvBlock(2, 128, (3, 3))
        self.conv3 = ConvBlock(3, 256, (3, 3))
        self.conv4 = ConvBlock(3, 512, (3, 3))
        self.conv5 = ConvBlock(3, 512, (3, 3))

    def call(self, input_tensor):

        img_input, f1 = self.conv1(input_tensor)
        _, f2 = self.conv2(f1)
        _, f3 = self.conv3(f2)
        _, f4 = self.conv4(f3)
        _, f5 = self.conv5(f4) 

        return img_input, [f1, f2, f3, f4, f5] 

        