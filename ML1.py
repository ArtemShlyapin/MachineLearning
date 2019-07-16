import cv2
import numpy as np
import os


W = np.zeros((65536, 1))
b = 1


def read_files(X, Y, path, ans):
    files = os.listdir(path)
    for name in files:
        img = cv2.imread(path + '/' + name, 0)
        if img.shape != 0:
            img = cv2.resize(img, (256, 256))
            vect = img.reshape(1, 256 ** 2)
            vect = vect / 255.
            X = vect if (X is None) else np.vstack((X, vect)) 
            Y = np.append(Y, ans)
    return X, Y

def sigmoid(p):
    nz = np.array([])
    for i in p:
        nz = np.append(nz, (1.0 / (1.0 + np.exp(-i))) )
    return nz

def propagate(W, b, X, Y):
    z = np.array([])
    WX = np.array([])
    A = np.array([])
    L = np.array([])
    dw = np.zeros(65536)
    db, J = 0, 0

    WX = np.dot(X, W)
    WX += b

    '''
    for i in WX:
        z = np.append(z, i + b)

    
    for j, i in enumerate(X):
        #print("Raschet sigmoidi", j)
        WX = np.append(WX, 0)
        for e in i:
            WX[j] += W[j] * e
        #print(WX[j])
        z = np.append(z, (WX[j] + b))
    '''
    A = np.append(A, sigmoid(WX))
    #print(A)
    for i, a in enumerate(A):
        #print("J")
        J += (-Y[i]*np.log(a) - (1.0 - Y[i])*np.log(1.0 - a)) / 42.0
        #print('a = ', a)
        #print('1.0 - a = ', 1.0 - a)
    #print(J)
    '''
    for e, i in enumerate(X):
        #print("dw")
        for u, j in enumerate(i):
            dw[u] += j * (A[e] - Y[e]) / 42.0
    
    A_Y = np.array([])
    for i, j in enumerate(Y)
        A_Y = np.append(A_Y, A[i] - j)
    '''

    A_Y = A - Y
    dw = np.dot(A_Y, X)
    dw = dw / 42.0
    
    for j, i in enumerate(Y):
        #print("db")
        db += A[j] + i
    db = db / 42.0
    
    #print(len(dw))

    grads = {"dw": dw, "db": db}

    return grads, J

#propagate(W, b, X, Y)


def optimize(W, b, X, Y, num_iterations, l_l, print_cost = False):

    costs = []
    
    for i in range(num_iterations):
        
        
        
        grads, cost = propagate(W, b, X, Y)
        
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        ### START CODE HERE ###
        l = len(W)
        for j in range(l - 1):
            W[j] = W[j] - l_l*dw[j]    
        b = b - l_l*db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 10 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": W,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(W, b, X):

    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)
    

    WX = np.dot(X, W)
    WX += b
    A = sigmoid(WX)
    
    for i in range(A.shape[1]):
        if A[i] > 0.7:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0
    
    return Y_predict


X = None
Y = np.array([])
path = '/home/artem/ML/Lesson_1/logloss_1'
X, Y = read_files(X, Y, path, 1)

optimize(W, b, X, Y, 100, 0.01, True)

X = None
Y = np.array([])
path = '/home/artem/ML/Lesson_1/logloss_0'
X, Y = read_files(X, Y, path, 0)

optimize(W, b, X, Y, 100, 0.01, True)

X = None
Y = np.array([])
path = '/home/artem/ML/Lesson_1/exp'
X, Y = read_files(X, Y, path, 1)

print(predict(W, b, X))