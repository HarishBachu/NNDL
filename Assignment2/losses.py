import numpy as np 

##### Regressive Loss Functions ##### 

def MSE(y_true, y_pred):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred) 

    diff = y_true - y_pred 
    sq = np.square(diff)

    return sq.mean() 


def MAE(y_true, y_pred):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred)

    diff = y_true - y_pred 
    ab = np.abs(diff)

    return ab.mean()

def MBE(y_true, y_pred): 
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred)

    diff = y_true - y_pred  
    return diff.mean()  

def Huber(y_true, y_pred, delta=1.0): 
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred)

    diff = y_true - y_pred
    norm = np.linalg.norm(diff)
    if norm <= delta: 
        return 0.5*norm
    else:
        return delta*np.linalg.norm(diff - 0.5*delta, ord = 1)

def epsilonHingeLoss(y_true, y_pred, epsilon=0.2):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred) 

    diff = y_true - y_pred - epsilon
    loss = diff * (diff > 0.0)
    return loss.mean()

def squaredEpsilonHingeLoss(y_true, y_pred, epsilon=0.2):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred)

    diff = np.linalg.norm(y_true - y_pred, ord=2) - epsilon**2
    loss = diff * (diff > 0.0)
    return loss.mean()/2


##### Classification Loss Functions ##### 

def binaryCrossEntropyLoss(y_true, y_pred):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred) 
    return -(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred)).mean()

def jaccardLoss(y_true, y_pred, epsilon=1e-6):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred)

    intersection = y_true.dot(y_pred).sum() 
    total = (y_true + y_pred).sum() 
    union = intersection - total 

    return 1 - (intersection + epsilon)/(union + epsilon) 

def diceLoss(y_true, y_pred, epsilon=1e-6):
    if(type(y_true) != np.ndarray):
        y_true = np.array(y_true) 
    if(type(y_pred) != np.ndarray):
        y_pred = np.array(y_pred)

    intersection = y_true.dot(y_pred).sum() 
    return 1 - (2*intersection + epsilon)/(y_true.sum() + y_pred.sum() - epsilon)


