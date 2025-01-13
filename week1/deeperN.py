import numpy as np

def main():
    layer_dims = [2, 4, 3, 1]
    A= np.array([[2.0],[3.0]]).reshape(-1, 1)
    Y= np.array([[1.0]]).reshape(-1, 1)
    p = initialize_parameters_deep(layer_dims)
    # 循环
    AL, caches = model_forward(A, p)
    grads = L_model_backward(AL, caches, Y)
    parameters = update_parameters(p, grads, 0.1)
    p = parameters
    '''c = compute_cost(AL, Y)'''

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for i in range(L):
        parameters["W"+str(i+1)] -= learning_rate*grads["dW"+str(i+1)]
        parameters["b"+str(i+1)] -= learning_rate*grads["db"+str(i+1)]
    return parameters

def L_model_backward(AL, caches, Y):
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads = {}
    grads["dA"+str(len(caches))], grads["dW"+str(len(caches))], grads["db"+str(len(caches))] = linear_activation_backward(dAL, caches[len(caches)-1], activation='sigmoid')
    l = range(len(caches)-1)
    for i in reversed(l):
        grads["dA"+str(i+1)], grads["dW"+str(i+1)], grads["db"+str(i+1)] = linear_activation_backward(grads["dA"+str(i+2)], caches[i], activation='reLu')
    return grads

# 反向传播
def linear_backward(dZ, cache):
    A_, W, b = cache
    m = W.shape[0]
    dW = np.dot(dZ, A_.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_ = np.dot(W.T, dZ)
    return dA_, dW, db

# sigmoid, reLu backward
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    a = 1/(1+np.exp(-Z))
    dZ =  dA * a * (1-a)
    return dZ

def reLu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation=='sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_, dW, db = linear_backward(dZ, linear_cache)
    if activation=='reLu':
        dZ = reLu_backward(dA, activation_cache)
        dA_, dW, db = linear_backward(dZ, linear_cache)
    return dA_, dW, db


# 初始化参数，两层网络
def initialize_parameter(x_n, x_h, x_y):
    W1 = np.random.randn(x_h, x_n) * 0.01
    b1 = np.zeros((x_h, 1))
    W2 = np.random.randn(x_y, x_h) * 0.01
    b2 = np.zeros((x_y, 1))
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    return parameters

# 初始化参数，多层网络
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

# 线性向前传播
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

# 单层网络输出
def linear_activation_forward(A_pre, W, b, activation):
    Z, linear_cache = linear_forward(A_pre, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    if activation == 'reLu':
        A, activation_cache = reLu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

# 多层网络输出
def model_forward(A_, parameters):
    caches = []
    L = len(parameters) // 2
    for i in range(1, L):
        A = A_
        A_, cache= linear_activation_forward(A, parameters["W"+str(i)], parameters["b"+str(i)], 'reLu')
        caches.append(cache)
    A = A_
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], 'sigmoid')
    caches.append(cache)
    return AL, caches
    
def sigmoid(Z):
    A = 1 + np.exp(-Z)
    cache = Z
    return 1/A, cache

def reLu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def compute_cost(AL, Y):
    m = AL.shape[0]
    cost = np.sum(np.multiply(Y, -np.log(AL+0.0001)) + np.multiply((1-Y), -np.log(1-AL+0.0001)))/m
    cost = np.squeeze(cost)
    return cost
main()