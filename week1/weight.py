import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from deeperN import *
def main():
    train_X, train_Y, test_X, test_Y = load_dataset(is_plot=False)
    p = model(train_X, train_Y, initialization = "he")

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_polt=True):
    m, n = X.shape
    layer_dim = [m, 10, 5, 1]
    costs = []
    if initialization == 'zeros':
        parameters = initialize_parameters_zero(layer_dim)
    if initialization == 'random':
        parameters = initialize_parameters_random(layer_dim)
    if initialization == 'he':
        parameters = initialize_parameters_he(layer_dim)
    for i in range(num_iterations):
        AL, cache = model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, cache, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 1000 == 0:
            costs.append(cost)
            #打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))
    return parameters
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
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

main()