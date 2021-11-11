import numpy as np 
import tensorflow as tf 

from tensorflow.keras import models, layers 

class FCN8(models.Model):
    def __init__(self):
        super(FCN8, self).__init__() 

        