import numpy as np
import sklearn
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from deeperN import *
def main():
    train_X, train_Y, test_X, test_Y = load_dataset(is_plot=False)
    p = model(X = train_X, Y = train_Y, lambd=0.7, keep_prob=1, num_iterations=30000)

def model_forward_dropout(X, parameters, keep_prob):
    np.random.seed(1)
    para = {}
    caches = []
    L = len(parameters)//2
    for i in range(1, L):
        para["Z"+str(i)] = np.dot(parameters["W"+str(i)], X) + parameters["b"+str(i)]
        para["A"+str(i)], _ = reLu(para["Z"+str(i)])
        m, n = para["A"+str(i)].shape
        para["D"+str(i)] = np.random.randn(m, n)
        para["D"+str(i)] = para["D"+str(i)] < keep_prob
        para["A"+str(i)] *= para["D"+str(i)]
        para["A"+str(i)] /= keep_prob
        X = para["A"+str(i)]
        caches.append((para["Z"+str(i)], para["D"+str(i)], para["A"+str(i)], parameters["W"+str(i)], parameters["b"+str(i)]))
    para["Z"+str(L)] = np.dot(parameters["W"+str(L)], X) + parameters["b"+str(L)]
    para["A"+str(L)], _ = sigmoid(para["Z"+str(L)])
    caches.append((para["Z"+str(L)], para["A"+str(L)], parameters["W"+str(L)], parameters["b"+str(L)]))
    return para["A"+str(L)], caches

def compute_cost_regularization(AL, lambd, parameters, Y):
    m = Y.shape[1] # AL.shape[0]
    regularization = 0
    cost = np.sum(np.multiply(Y, -np.log(AL+0.00001)) + np.multiply((1-Y), -np.log(1-AL+0.00001)))
    L = len(parameters)//2
    for i in range(1, L+1):
        regularization += lambd*np.sum(parameters["W"+str(i)]**2)/(2*m)
    cost += regularization
    return cost

def model_backward_regularization(AL, Y, cache, lambd):
    m = AL.shape[1]
    grads = {}
    L = len(cache)
    grads["dZ"+str(L)] = AL - Y
    A = cache[L-2][1]
    # ** backward without regularization still has some problems
    grads["dW"+str(L)] = np.dot(grads["dZ"+str(L)], A.T)/m
    grads["db"+str(L)] = np.sum(grads["dZ"+str(L)], axis=1, keepdims=True)/m
    grads["dA"+str(L-1)] = np.dot(grads["dW"+str(L)].T, grads["dZ"+str(L)])
    l = range(1, len(cache))
    for i in reversed(l):
        grads["dZ"+str(i)] = np.multiply(grads["dA"+str(i)],np.int64(cache[i-1][2] > 0))
        A = cache[i-2][1]
        grads["dW"+str(i)] = np.dot(grads["dZ"+str(i)], A.T)/m +((lambd * cache[i-1][3]) / m)
        grads["db"+str(i)] = np.sum(grads["dZ"+str(i)], axis=1, keepdims=True)/m
        grads["dA"+str(i-1)] = np.dot(grads["dW"+str(i)].T, grads["dZ"+str(i)])
    return grads

def model(X,Y,learning_rate=0.3,num_iterations=1000, print_cost=True, is_polt=True, lambd=0, keep_prob=1):
    m, n = X.shape
    grads = {}
    costs = []
    layer_dim = [m, 20, 3, 1]
    parameters = initialize_parameters_he(layer_dim)
    for i in range(num_iterations):
        # keep_prob
        if keep_prob == 1:
            AL, cache = model_forward(X, parameters)
        if keep_prob < 1:
            AL, cache = model_forward_dropout(X, parameters, keep_prob)
        # 正则化
        if lambd == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_regularization(AL, lambd, parameters, Y)
        # 反向
        if (keep_prob == 1 and lambd == 0):
            grads = L_model_backward(AL, cache, Y)
        elif lambd != 0:
            grads = model_backward_regularization(AL, Y, cache, lambd)
        elif keep_prob < 1:
            grads = model_backward_dropout(AL, Y, cache, keep_prob)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        if i % 1000 == 0:
            ## 记录成本
            costs.append(cost)
            print("第" + str(i) + "次迭代，成本值为：" + str(cost))
    return parameters

def model_backward_dropout(AL, Y, cache, keep_prob):
    m = AL.shape[1]
    grads = {}
    L = len(cache)

    grads["dZ"+str(L)] = sigmoid_backward(AL, cache[L-1][0]) # cache Z
    # Z2 = AL-Y
    A = cache[L-2][1]
    grads["dW"+str(L)] = np.dot(grads["dZ"+str(L)], A.T)/m
    grads["db"+str(L)] = np.sum(grads["dZ"+str(L)], axis=1, keepdims=True)/m
    grads["dA"+str(L-1)] = np.dot(grads["dW"+str(L)].T, grads["dZ"+str(L)])
    l = range(1, len(cache))
    for i in reversed(l):
        grads["dA"+str(i)] *= cache[i-1][1]
        grads["dA"+str(i)] /= keep_prob
        
        grads["dZ"+str(i)] = np.multiply(grads["dA"+str(i)],np.int64(cache[i-1][2] > 0))
        # Z = reLu_backward(grads["dA"+str(i)], cache[i-1][0]) 相同
        A = cache[i-2][1]
        grads["dW"+str(i)] = np.dot(grads["dZ"+str(i)], A.T)/m
        grads["db"+str(i)] = np.sum(grads["dZ"+str(i)], axis=1, keepdims=True)/m
        grads["dA"+str(i-1)] = np.dot(grads["dW"+str(i)].T, grads["dZ"+str(i)])
    
    return grads

def initialize_parameters_he(layer_dims):
    np.random.seed(3) 
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def initialize_parameters_zero(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters["W"+str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def initialize_parameters_random(layer_dims):
    np.random.seed(3) 
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def load_dataset(is_plot=True):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_blobs(n_samples=300, centers=[[-1,-1], [1,1]], cluster_std=[4, 5], n_features=2, random_state=1)
    train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=100)
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

main()