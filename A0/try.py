import numpy as np
from numpy import ma
import math
from sklearn.preprocessing import OneHotEncoder
dic = {0:[1,2],1:[0, 3]}

a = np.array([[2,3,4,5,6], [1, 1, 1, 1, 1], [2,1,4,3,6], [4,4,4,4,4]])
# print(a[0] * a[0:2])
b = np.array([0, 1, 1, 2])
a1 = np.array([[0,3,4,8,6], [1, 0, 1, 2, 5], [2,1,5,3,0], [4,1,4,1,1]])
b1 = np.array([1, 2, 1, 0])

'''
k nearest 识别

k = 2
recognization = np.zeros_like(b1)
dist = np.zeros(shape=(len(a), len(a)))
ks = np.zeros(shape=(len(a), k))
for i, c1 in enumerate(a):
    for j, c2 in enumerate(a):
        dist[i, j] = math.dist(c1, c2)
m_matrix = ma.masked_array(dist, mask=np.identity(dist.shape[0]))
print(m_matrix)
for i, d in enumerate(m_matrix):
    ks[i] = m_matrix[i].argsort()[:k]
    recognization[i] = np.argmax(np.bincount(b[ks[i].astype(int)]))
    print(recognization[i])
'''
'''
center识别
recognization = np.zeros_like(b1)
for i, t in enumerate(a1): 
    distance = []
    for j, c in enumerate(a):
        distance.append(math.dist(t, c))
    print(distance)
    recognization[i] = np.argmin(np.array(distance))
accuracy = np.sum(recognization != b1)/len(b1)
print(accuracy)
'''


'''center
number = dict.fromkeys(b[:].astype(int))
center = np.zeros(shape=(3, a.shape[1]))
for i in range(3):
    l = []
    for j, character in enumerate(b):
        if character == i:
            l.append(j)
    number[i] = l
print(number)

for k in range(3):
    center[k] = np.mean(a[number[k]], axis=0)
print(center)'''


'''np.random.seed(1)
b = np.random.rand(1, 10)
W = np.random.rand(2, 10)
print('W', W)
X = np.array([[1, 2], [8, 4], [5, 6]])
print('R', X @ W + b)

X = np.insert(X, X.shape[1], 1, axis=1)
print('nX', X)
W = np.insert(W, W.shape[0], b, axis=0)
print('nW', W)

output = X @ W
print(output)

digit = np.argmax(X, axis=1)
print(digit)'''


labels = np.array([0, 1, 2, 1, 0]).reshape(-1, 1)

# 创建OneHotEncoder对象
encoder = OneHotEncoder(sparse_output=False)

# 进行编码
one_hot_encoded = encoder.fit_transform(labels)
print(one_hot_encoded)
print(np.eye(10)[labels.flatten()])