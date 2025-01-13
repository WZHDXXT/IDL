import numpy as np
from deeperN import *
def main():
    dim_layers = [2, 3, 2, 1]
    X = np.array([[0.2],[0.3]])
    Y = np.array([[1]]).reshape(-1, 1)
    parameters = initialize_parameters_deep(dim_layers)
    AL, cache = model_forward(X, parameters)
    grads = L_model_backward(AL, cache, Y)
    d =  gradient_check_n(parameters, grads, X, Y, epsilon=1e-7)
    print(d)

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    parameters_values, keys = dictionary_to_vector(parameters)
    grads = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))
    # 计算gradapprox
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        J_plus[i], cache = model_forward(X, vector_to_dictionary(thetaplus)) 

        thetamin = np.copy(parameters_values)
        thetamin[i][0] -= epsilon
        J_minus[i], cache = model_forward(X, vector_to_dictionary(thetamin))
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    numerator = np.linalg.norm(grads - gradapprox)
    denominator = np.linalg.norm(grads) + np.linalg.norm(gradapprox) 
    difference = numerator/denominator

    return difference

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:6].reshape((3,2))
    parameters["b1"] = theta[6:9].reshape((3,1))
    parameters["W2"] = theta[9:15].reshape((2,3))
    parameters["b2"] = theta[15:17].reshape((2,1))
    parameters["W3"] = theta[17:19].reshape((1,2))
    parameters["b3"] = theta[19:20].reshape((1,1))
 
    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
 
    return theta

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            # 数组堆叠
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
 
    return theta, keys
main()