import numpy as np 

##### Activations #####

def sigmoid(x):             #Sigmoid Activation
    return 1/(1 + np.exp(-x))

def tanh(x):                #Tanh Activation 
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def softsign(x):            #SoftSign Activation
    return x/(1 + abs(x))

def ReLU(x):                #Relu Activation
    return x * (x > 0.0)

def softmax(x):             #Softmax Activation
    e_x = np.exp(x)
    return e_x/e_x.sum()


##### Derivatives #####

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def d_tanh(x):
    return 1 - tanh(x)**2  

def d_softsign(x):
    return 1/((1 + abs(x))**2)

def d_ReLU(x):
    return 1 * (x > 0.0)

    