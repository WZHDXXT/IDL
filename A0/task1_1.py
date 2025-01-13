from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.ma as ma
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

N = 10
def main():
    train_in, test_in, train_out, test_out = load_data()
    center = center_compute(train_in, train_out)
     
    r2, a2 = digit_recognization(center, test_in, test_out)
    print(a2)

def load_data():
    train_in = pd.read_csv('./data/train_in.csv', header=None)
    test_in = pd.read_csv('./data/test_in.csv', header=None)
    train_out = pd.read_csv('./data/train_out.csv', header=None)
    test_out = pd.read_csv('./data/test_out.csv', header=None)
    return np.array(train_in), np.array(test_in), np.array(train_out), np.array(test_out)

def center_compute(train_in, train_out):
    number = dict.fromkeys(train_out[:].reshape(1, -1)[0].astype(int), [0])
    center = np.zeros(shape=(N, train_in.shape[1]))
    for i in number.keys():
        l = []
        for j, character in enumerate(train_out):
            if character == i:
                l.append(j)
        number[i] = l
    for d in number.keys():
        center[d] = np.mean(train_in[number[d]], axis=0)
    return center

def digit_recognization(center, test_in, test_out):
    recognization = np.zeros_like(test_out)
    for i, t in enumerate(test_in):
        distance = []
        for j, c in enumerate(center):
            distance.append(np.linalg.norm(t - c))
        recognization[i] = np.argmin(np.array(distance))
    accuracy = np.sum(recognization == test_out, dtype=float)/len(test_out)
    return recognization, accuracy

def dis_between_centers(center):
    dis_matrix = np.zeros(shape=(N, N))
    center_copy = np.copy(center)
    for i, ci in enumerate(center_copy):
        for j, cj in enumerate(center):
            if i != j:
                dis_matrix[i][j] = np.linalg.norm(ci-cj)
    
    m_matrix = ma.masked_array(dis_matrix, mask=np.identity(dis_matrix.shape[0]))
    indice = np.where(m_matrix == np.min(m_matrix))
    return dis_matrix, indice
main()