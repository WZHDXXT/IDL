import numpy as np
import numpy.ma as ma
import pandas as pd

N = 10
def main():
    train_in, test_in, train_out, test_out = load_data()
    classer = Nearest_mean_classifier(train_in, test_in, train_out, test_out)
    print(classer.con_matrix)

def load_data():
    train_in = pd.read_csv('./data/train_in.csv', header=None)
    test_in = pd.read_csv('./data/test_in.csv', header=None)
    train_out = pd.read_csv('./data/train_out.csv', header=None)
    test_out = pd.read_csv('./data/test_out.csv', header=None)
    return np.array(train_in), np.array(test_in), np.array(train_out), np.array(test_out)

class Nearest_mean_classifier():
    def __init__(self, train_in, test_in, train_out, test_out):
        self.train_in = train_in
        self.test_in = test_in
        self.train_out = train_out
        self.test_out = test_out
        self.center = self.center_compute()
        self.train_recognization, self.train_accuracy = self.digit_recognization_train()
        self.test_recognization, self.test_accuracy = self.digit_recognization_test()
        self.dis_matrix, self.indice = self.dis_between_centers()
        self.con_matrix = self.confusion_matrix()
    def center_compute(self):
        number = dict.fromkeys(self.train_out[:].reshape(1, -1)[0].astype(int), [0])
        center = np.zeros(shape=(N, self.train_in.shape[1]))
        for i in number.keys():
            l = []
            for j, character in enumerate(self.train_out):
                if character == i:
                    l.append(j)
            number[i] = l
        for d in number.keys():
            center[d] = np.mean(self.train_in[number[d]], axis=0)
        return center

    def digit_recognization_train(self):
        recognization = np.zeros_like(self.train_out)
        for i, t in enumerate(self.train_in):
            distance = []
            for j, c in enumerate(self.center):
                distance.append(np.linalg.norm(t - c))
            recognization[i] = np.argmin(np.array(distance))
        accuracy = np.sum(recognization == self.train_out, dtype=float)/len(self.train_out)
        return recognization, accuracy
    
    def digit_recognization_test(self):
        recognization = np.zeros_like(self.test_out)
        for i, t in enumerate(self.test_in):
            distance = []
            for j, c in enumerate(self.center):
                distance.append(np.linalg.norm(t - c))
            recognization[i] = np.argmin(np.array(distance))
        accuracy = np.sum(recognization == self.test_out, dtype=float)/len(self.test_out)
        return recognization, accuracy
    
    def dis_between_centers(self):
        dis_matrix = np.zeros(shape=(N, N))
        center_copy = np.copy(self.center)
        for i, ci in enumerate(center_copy):
            for j, cj in enumerate(self.center):
                if i != j:
                    dis_matrix[i][j] = np.linalg.norm(ci-cj)
        
        m_matrix = ma.masked_array(dis_matrix, mask=np.identity(dis_matrix.shape[0]))
        indice = np.where(m_matrix == np.min(m_matrix))
        return dis_matrix, indice
    
    def confusion_matrix(self):
        con_matrix = np.zeros(shape=(N, N))
        for i, ci in enumerate(self.test_out):
            if ci[0] != self.test_recognization[i][0]:
                con_matrix[ci[0], self.test_recognization[i][0]] += 1
        for j, cj in enumerate(self.train_out):
            if cj[0] != self.train_recognization[j][0]:
                con_matrix[cj[0], self.train_recognization[j][0]] += 1
        return con_matrix
main()