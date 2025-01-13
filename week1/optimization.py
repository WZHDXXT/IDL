from deeperN import *
import numpy as np
import sklearn
import sklearn.datasets
import math

def main():
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) 
    m, n = train_X.shape
    train_X = train_X.reshape(n, m)
    train_Y = train_Y.reshape(1, m)
    model(train_X, train_Y)

def model(X, Y, iters_num=10000, layers_dims=[1,3,4], optimizer='gd', learning_rate=0.0007, mini_batch_size=64,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    parameters = initialize_parameters_he(layers_dims)
    # 选择优化器并初始化参数
    if optimizer == 'momentun':
        v = initialize_velocity(parameters)
    if optimizer == 'adam':
        v, s = initialize_adam(parameters)
    if optimizer == 'gd':
        pass
    else:
        exit(1)
    # start
    for i in range(iters_num):
        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            A3, cache = model_forward(minibatch_X, parameters)
            cost = compute_cost(A3, minibatch_Y)
            grads = L_model_backward(A3, cache, minibatch_Y)
            if optimizer == 'momentun':
                parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
            if optimizer == 'adam':
                t += 1
                parameters, v, s = update_parameters_with_adam(parameters, v, s, grads, t, learning_rate, beta1, beta2, epsilon)
            if optimizer == 'gd':
                parameters= update_parameters(parameters, grads, learning_rate)
            else:
                exit(1)

        
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 打乱顺序
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    # 分割
    num_complete_minibatches = math.floor(m/mini_batch_size)
    # 返回剩下的小数
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return v

def initialize_parameters_he(layer_dims):
    np.random.seed(3) 
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i], 1))
    return parameters

def update_parameters_with_momentun(parameters, grads, v, beta, learning_rate):
    L = len(parameters)//2
    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW"+str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db"+str(l+1)]
        parameters["dW"+str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["db"+str(l+1)] -= learning_rate * v["db" + str(l+1)]
    return parameters, v

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return (v,s)

def update_parameters_with_adam(parameters, v, s, grads, t, learning_rate, beta1, beta2, epsilon):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}
    for l in L:
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-np.power(beta1, t))
        
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.square(grads["db" + str(l+1)])
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-np.power(beta2, t))

        parameters["dW" + str(l+1)] -= learning_rate * v_corrected["dW" + str(l+1)]/np.square(s_corrected["dW" + str(l+1)]+epsilon)
        parameters["db" + str(l+1)] -= learning_rate * v_corrected["dW" + str(l+1)]/np.square(s_corrected["dW" + str(l+1)]+epsilon)
    return (parameters, v, s)


main()



