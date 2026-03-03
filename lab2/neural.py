import numpy as np

class Perceptron:
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
        
        self.Win1 = np.zeros((1+inputSize,hiddenSize1))
        self.Win2 = np.zeros((1+hiddenSize1,hiddenSize2))
        self.Win1[0,:] = (np.random.randint(0, 3, size = (hiddenSize1)))
        self.Win1[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSize1)))
        
        self.Win2[0,:] = (np.random.randint(0, 3, size = (hiddenSize2)))
        self.Win2[1:,:] = (np.random.randint(-1, 2, size = (hiddenSize1,hiddenSize2)))
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSize2,outputSize)).astype(np.float64)

    def predict(self, Xp):
        hidden_predict1 = np.where((np.dot(Xp, self.Win1[1:,:]) + self.Win1[0,:]) >= 0.0, 1, -1).astype(np.float64)
        hidden_predict2 = np.where((np.dot(hidden_predict1, self.Win2[1:,:]) + self.Win2[0,:]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict2, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        return out, hidden_predict2

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self

