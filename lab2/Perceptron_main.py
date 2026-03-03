import pandas as pd
import numpy as np
from lab2.neural import Perceptron


df = pd.read_csv('lab2/data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values
max_acc = 0
epoch = 0

#%%
inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSize1 = 10 # задаем число нейронов скрытого (А) слоя 
hiddenSize2 = 100
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

#%%
NN = Perceptron(inputSize, hiddenSize1,hiddenSize2, outputSize)

NN.train(X, y, n_iter=5, eta = 0.01)

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
out, hidden_predict = NN.predict(X)

sum_err = abs(int(sum(out-y.reshape(-1, 1))[0]/2))
acc = 100 - sum_err / len(df) * 100
max_acc = acc if acc > max_acc else max_acc
epoch += 1
print(f"{epoch}\nВсего ошибок: {sum_err:3} | Точность: {acc:3.2f} %\nМаксимальная точность: {max_acc:3.2f} %")
