import numpy as np

bias = np.array([1])
def main():
    iteration = 2
    type = input("Input type of activation function:")
    np.random.seed(1)
    W = np.random.rand(1, 9)
    # W = np.zeros(shape=(1, 9))
    W = W.reshape(3, 3)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.longdouble)
    # 4*3
    W1 = W[:, :2]
    W2 = W[:, 2].reshape(-1, 1)
    targets = np.array([0, 1, 1, 0], dtype=np.longdouble).reshape(-1, 1)
    match type:
            case 'sigmoid':
                activation = sigmoid
                derivative = sigmoid_derivative
            case 'reLu':
                activation = reLu
                derivative = rel_derivative
            case 'tanh':
                activation = tanh
                derivative = tanh_derivative
    for i in range(iteration):
        x1, output1 = xor_net(X, W1, activation)
        x2, output2 = xor_net(output1, W2, activation)
        x1.astype(np.longdouble)
        x2.astype(np.longdouble)
        output2_ = np.copy(output2)
        output2_ = np.where(output2_ > 0.5, 1, 0)
        mse = error_function(X, output2, targets)
        miss_classify = np.sum(output2_ != targets)
        alpha = 0.8

        # backpropagation
        '''dZ2 = (output2.reshape(-1, 1) - targets) * sigmoid_derivative(x2.reshape(-1, 1))
        dW2 = back_prop(output1, dZ2)
        dZ1 = dZ2 @ (W2[:2].reshape(2, 1).T) * sigmoid_derivative(x1)
        dW1 = back_prop(X, dZ1)
        W2 -= alpha * dW2
        W1 -= alpha * dW1'''

        W1, W2 = backward_propagation(X, targets, x1, x2, output1, output2, W1, W2, alpha, derivative)
        if i % 10 == 0:
            print("The " + str(i) + " th mse is " + str(mse))
            print("The " + str(i) + " th miss classified number is " + str(miss_classify))

def backward_propagation(X, targets, x1, x2, output1, output2, W1, W2, alpha, derivative):
    m1 = X.shape[0]
    m2 = output1.shape[0]
    # 4*1
    dZ2 = (output2.reshape(-1, 1) - targets) * derivative(output2.reshape(-1, 1))
    output1_copy = np.copy(output1)
    output1_copy1 = np.insert(output1_copy, 0, bias, axis=1)# 4*3
    dW2 = (1 / m1) * (output1_copy1.T @ dZ2) # 3*1
    dA1 = dZ2 @ (W2[:2].reshape(2, 1).T)# 4*1 1*2 = 4*2
    dZ1 = dA1 * derivative(output1)# 4*2
    X_copy = np.copy(X)
    X_copy1 = np.insert(X_copy, 0, bias, axis=1)# 4*3
    dW1 = 1. / m2 * (X_copy1.T @ dZ1) # 3*2
    W2 -= alpha * dW2
    W1 -= alpha * dW1
    return W1, W2

# XOR neural network
def xor_net(X, W, activation):    
    X_input = np.insert(X, 0, bias, axis=1)
    output1 = X_input @ W
    layer_out = activation(output1)
    return output1, layer_out

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def reLu(x):
    return np.maximum(0, x)

def rel_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Mean squared error function
def error_function(X, output, targets):
    # Define the 4 possible inputs and corresponding target outputs
    total_error = np.sum((output - targets) ** 2)
    mse = total_error / len(X)
    return mse

# extra
def back_prop(X, dZ):
    m = X.shape[0]
    X_copy = np.copy(X)
    X_copy1 = np.insert(X_copy, 0, bias, axis=1)
    dW = (1/m) * (X_copy1.T @ dZ)
    return dW

def backward_propagation_tanh(X, targets, x1, x2, output1, output2, W1, W2, alpha):
    m1 = X.shape[0]
    m2 = output1.shape[0]
    # 4*1
    dZ2 = (output2.reshape(-1, 1) - targets) * tanh_derivative(x2.reshape(-1, 1))
    output1_copy = np.copy(output1)
    output1_copy1 = np.insert(output1_copy, 0, bias, axis=1)# 4*3
    dW2 = (1 / m1) * (output1_copy1.T @ dZ2) # 3*1
    dA1 = dZ2 @ (W2[:2].reshape(2, 1).T)# 4*1 1*2 = 4*2
    
    dZ1 = dA1 * tanh_derivative(x1)# 4*2
    X_copy = np.copy(X)
    X_copy1 = np.insert(X_copy, 0, bias, axis=1)# 4*3
    dW1 = 1. / m2 * (X_copy1.T @ dZ1) # 3*2
    W2 -= alpha * dW2
    W1 -= alpha * dW1
    return W1, W2

def backward_propagation_reLu(X, targets, x1, x2, output1, output2, W1, W2, alpha):
    m1 = X.shape[0]
    m2 = output1.shape[0]
    # 4*1
    dZ2 = (output2.reshape(-1, 1) - targets) * rel_derivative(x2.reshape(-1, 1))

    output1_copy = np.copy(output1)
    output1_copy1 = np.insert(output1_copy, 0, bias, axis=1)# 4*3
    dW2 = (1 / m1) * (output1_copy1.T @ dZ2) # 3*1
    dA1 = dZ2 @ (W2[:2].reshape(2, 1).T)# 4*1 1*2 = 4*2
    
    dZ1 = dA1 * rel_derivative(x1)# 4*2
    X_copy = np.copy(X)
    X_copy1 = np.insert(X_copy, 0, bias, axis=1)# 4*3
    dW1 = 1. / m2 * (X_copy1.T @ dZ1) # 3*2
    W2 -= alpha * dW2
    W1 -= alpha * dW1
    return W1, W2

def backward_propagation_sigmoid(X, targets, x1, x2, output1, output2, W1, W2, alpha):
    m1 = X.shape[0]
    m2 = output1.shape[0]
    # 4*1
    dZ2 = (output2.reshape(-1, 1) - targets) * sigmoid_derivative(x2.reshape(-1, 1))
    output1_copy = np.copy(output1)
    output1_copy1 = np.insert(output1_copy, 0, bias, axis=1)# 4*3
    dW2 = (1 / m1) * (output1_copy1.T @ dZ2) # 3*1
    dA1 = dZ2 @ (W2[:2].reshape(2, 1).T)# 4*1 1*2 = 4*2
    
    dZ1 = dA1 * sigmoid_derivative(x1)# 4*2
    X_copy = np.copy(X)
    X_copy1 = np.insert(X_copy, 0, bias, axis=1)# 4*3
    dW1 = 1. / m2 * (X_copy1.T @ dZ1) # 3*2
    W2 -= alpha * dW2
    W1 -= alpha * dW1
    return W1, W2


def backward_propagation_reLu2(X, targets, output1, output2, W1, W2, alpha):
    m1 = X.shape[0]
    m2 = output1.shape[0]
    # 4*1
    dZ2 = (output2.reshape(-1, 1) - targets)
    output1_copy = np.copy(output1)
    output1_copy1 = np.insert(output1_copy, 0, bias, axis=1)# 4*3
    dZ2[(output1_copy1 @ W2).flatten() <= 0] = 0
    dW2 = (1 / m1) * (output1_copy1.T @ dZ2) # 3*1
    dA1 = dZ2 @ (W2[:2].reshape(2, 1).T)# 4*1 1*2 = 4*2
    
    dZ1 = dA1# 4*2
    X_copy = np.copy(X)
    X_copy1 = np.insert(X_copy, 0, bias, axis=1)# 4*3
    dZ1[(X_copy1@W1) <= 0] = 0
    dW1 = 1. / m2 * (X_copy1.T @ dZ1) # 3*2
    W2 -= alpha * dW2
    W1 -= alpha * dW1
    return W1, W2
main()