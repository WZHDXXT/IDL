import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data():
    train_in = pd.read_csv('./data/train_in.csv', header=None)
    test_in = pd.read_csv('./data/test_in.csv', header=None)
    train_out = pd.read_csv('./data/train_out.csv', header=None)
    test_out = pd.read_csv('./data/test_out.csv', header=None)
    return np.array(train_in, dtype=np.longdouble), np.array(test_in, dtype=np.longdouble), np.array(train_out, dtype=np.longdouble), np.array(test_out, dtype=np.longdouble)

def softmax(X):
    exp_a = np.exp(X - np.max(X, axis=1, keepdims=True))
    exp_sum = np.sum(exp_a, axis=1).reshape(-1, 1)
    softmax_output = exp_a / exp_sum
    return softmax_output

def multi_class_1(X, W, b):
    X = np.insert(X, X.shape[1], 1, axis=1)
    W = np.insert(W, W.shape[0], b, axis=0)
    output = X @ W # 170000*10
    digit = np.argmax(output, axis=1)
    return digit

def one_hot_encoding(X):
    X = X.reshape(-1, 1)
    encoder_train = OneHotEncoder(sparse_output=False)
    one_hot_X = encoder_train.fit_transform(X)
    return one_hot_X

def loss_function_1(X, Y, W, b):
    m, n = X.shape
    Y_h = X @ W + b
    loss = np.sum((Y_h - Y)**2)/m
    return loss

def gradient_descent_1(X, Y, W, b, learning_rate):
    m, n = X.shape
    Y_h = X @ W + b
        
    err = Y_h - Y
    dW = 2 * (X.T @ err) / m
    db = 2 * np.sum(err)/m
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b
train_in, test_in, train_out, test_out = load_data()
epoch = 100
Phi1 = np.zeros((256, 10))
Phi0 = np.zeros((1, 10))
accuracy_train = []
accuracy_test = []
loss_train = []
loss_test = []
learning_rate = np.arange(0.0001, 0.009, 0.0001)
for alpha in learning_rate:
    one_hot_train = one_hot_encoding(train_out)
    one_hot_train.astype(np.longdouble)
    one_hot_test = one_hot_encoding(test_out)
    one_hot_test.astype(np.longdouble)
    for i in range(epoch):
        Phi1, Phi0 = gradient_descent_1(train_in, one_hot_train, Phi1, Phi0, alpha)

    loss_f = loss_function_1(train_in, one_hot_train, Phi1, Phi0)
    loss_train.append(loss_f)
    loss_t = loss_function_1(test_in, one_hot_test, Phi1, Phi0)
    loss_test.append(loss_t)

    digit_f = multi_class_1(train_in, Phi1, Phi0)
    accuracy = np.sum(digit_f.reshape(-1, 1) == train_out, dtype=float)/len(train_out)
    accuracy_train.append(accuracy)
    digit_t = multi_class_1(test_in, Phi1, Phi0)
    accuracy_t = np.sum(digit_t.reshape(-1, 1) == test_out, dtype=float)/len(test_out)
    accuracy_test.append(accuracy_t)

# %%
fig, ax = plt.subplots(1,2)
ax[0].set_title("loss")
ax[0].set_xlabel("learning_rate")
ax[0].plot(learning_rate, loss_train, color='b',)
ax[0].plot(learning_rate, loss_test, color='r')
ax[1].set_title("accuracy")
ax[1].set_xlabel("learning_rate")
ax[1].plot(learning_rate, accuracy_train, color='b', label='training set')
ax[1].plot(learning_rate, accuracy_test, color='r', label='test set')
plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.show()