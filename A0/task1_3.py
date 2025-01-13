import numpy as np
import pandas as pd
import numpy.ma as ma
from sklearn.neighbors import KNeighborsClassifier


N = 10
k = 5
def main():
    train_in, test_in, train_out, test_out = load_data()
    classer = K_Nearest_classifier(train_in, test_in, train_out, test_out, k)
    print(classer.con_matrix)
    print(classer.test_accuracy)
    
    '''knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_in, train_out)
    accuracy = knn.score(test_in, test_out)
    print(accuracy)'''

def load_data():
    train_in = pd.read_csv('./data/train_in.csv', header=None)
    test_in = pd.read_csv('./data/test_in.csv', header=None)
    train_out = pd.read_csv('./data/train_out.csv', header=None)
    test_out = pd.read_csv('./data/test_out.csv', header=None)
    return np.array(train_in), np.array(test_in), np.array(train_out), np.array(test_out)

class K_Nearest_classifier():
    def __init__(self, train_in, test_in, train_out, test_out, k):
        self.k = k
        self.train_in = train_in
        self.test_in = test_in
        self.train_out = train_out
        self.test_out = test_out
        self.train_recognization, self.train_accuracy = self.k_classifier_train()
        self.test_recognization, self.test_accuracy = self.k_classifier_test()
        self.con_matrix = self.confusion_matrix()
    
    def k_classifier_train(self):
        recognization = np.zeros_like(self.train_out)
        dist = np.zeros(shape=(len(self.train_in), len(self.train_in)))
        ks = dict.fromkeys(self.train_out[:].reshape(1, -1)[0].astype(int), [0])
        for i, ci in enumerate(self.train_in):
            for j, cj in enumerate(self.train_in):
                dist[i, j] = np.linalg.norm(ci-cj)
        m_matrix = ma.masked_array(dist, mask=np.identity(dist.shape[0]))
        for d, cj in enumerate(self.train_in):
            ks[d] = m_matrix[d].argsort()[:k]
            ks_d_array = np.array(ks[d])
            recognization[d] = np.argmax(np.bincount(self.train_out[ks_d_array].reshape(1, -1)[0]))
        accuracy = np.sum(recognization == self.train_out)/len(self.train_out)
        return recognization, accuracy
    
    def k_classifier_test(self):
        recognization = np.zeros_like(self.test_out)
        dist = np.zeros(shape=(len(self.test_in), len(self.train_in)))
        ks = dict.fromkeys(self.test_out[:].reshape(1, -1)[0].astype(int), [0])
        for i, ci in enumerate(self.test_in):
            for j, cj in enumerate(self.train_in):
                dist[i, j] = np.linalg.norm(ci-cj)
        for d, cj in enumerate(self.test_in):
            ks[d] = dist[d].argsort()[:k]
            ks_d_array = np.array(ks[d])
            recognization[d] = np.argmax(np.bincount(self.train_out[ks_d_array].reshape(1, -1)[0]))
        accuracy = np.sum(recognization == self.test_out)/len(self.test_out)
        return recognization, accuracy

    def confusion_matrix(self):
        con_matrix = np.zeros(shape=(N, N))
        for i, ci in enumerate(self.test_out):
            if ci != self.test_recognization[i]:
                con_matrix[int(ci), int(self.test_recognization[i])] += 1
        for j, cj in enumerate(self.train_out):
            if cj != self.train_recognization[j]:
                con_matrix[int(cj), int(self.train_recognization[j])] += 1
        return con_matrix
    
main()