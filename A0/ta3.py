import numpy as np

# 定义输入和标签
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 定义神经网络结构
input_size = 2
hidden_size = 2
output_size = 1

# 随机初始化权重
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# 定义sigmoid函数作为激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义前向传播过程
def forward(X):
    # 第一层：输入到隐藏层
    z1 = np.dot(X, W1)
    h1 = sigmoid(z1)

    # 第二层：隐藏层到输出
    z2 = np.dot(h1, W2)
    y_hat = sigmoid(z2)

    return h1, y_hat

# 定义损失函数
def loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

# 定义反向传播过程
def backward(X, y, h1, y_hat):
    # 第二层：输出到隐藏层
    delta2 = (y - y_hat) * y_hat * (1 - y_hat)
    dW2 = np.dot(h1.T, delta2)

    # 第一层：隐藏层到输入
    delta1 = np.dot(delta2, W2.T) * h1 * (1 - h1)
    dW1 = np.dot(X.T, delta1)

    return dW1, dW2

# 训练神经网络
learning_rate = 0.1
for i in range(5000):
    # 前向传播
    h1, y_hat = forward(X)

    # 计算损失函数
    l = loss(y, y_hat)

    # 反向传播
    dW1, dW2 = backward(X, y, h1, y_hat)

    # 更新权重
    W1 += learning_rate * dW1
    W2 += learning_rate * dW2

    # 每隔1000次迭代输出一次损失函数的值
    if i % 1000 == 0:
        print("Iteration {}: Loss = {}".format(i, l))

# 测试神经网络
_, y_pred = forward(X)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("Final predictions: ")
print(np.sum(y_pred==y))
