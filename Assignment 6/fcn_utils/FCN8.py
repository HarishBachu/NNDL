import numpy as np 
import tensorflow as tf 

from tensorflow.keras import models, layers 
from fcn_utils.encoder import * 


class FCN8(models.Model):
    def __init__(self, n_classes):
        super(FCN8, self).__init__() 

        self.encoder = Encoder()
        self. convDropLayers = models.Sequential([
            ConvDropout(), 
            ConvDropout(), 
            layers.Conv2D(n_classes, (1, 1), activation="relu")
        ])

        self.convCrop1 = ConvCrop(n_classes, kernelSize=(4, 4), strides=(2, 2))
        self.conv1 = layers.Conv2D(n_classes, (1, 1), activation="relu") 
        self.add1 = layers.Add() 

        self.convCrop2 = ConvCrop(n_classes, kernelSize=(4, 4), strides=(2, 2))
        self.conv2 = layers.Conv2D(n_classes, (1, 1), activation="relu")
        self.add2 = layers.Add() 

        self.convCrop3 = ConvCrop(n_classes, kernelSize=(16, 16), strides=(8, 8))

        self.out = layers.Softmax(axis=3) 

    def call(self, input_tensor):
        img_input, [_, _, f3, f4, f5] = self.encoder(input_tensor) 

        o = f5 
        o = self.convDropLayers(o)

        o = self.convCrop1(o)
        o2 = f4
        o2 = self.conv1(o2)

        o = self.add1([0, o2])
    
        o = self.convCrop2(o)
        o2 = f3 
        o2 = self.conv2(o2)

        o = self.add2([o2, o])

        o = self.convCrop3(o) 
        o = self.out(o)

        return o 

    

        