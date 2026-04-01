import pandas as pd
import numpy as np
from lab3.neural import MLP


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

iterations = 51
learning_rate = 0.1

net1 = MLP(inputSize, outputSize, learning_rate, hiddenSizes)
net2 = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

X = pd.DataFrame(X)
y = pd.DataFrame(y)
Xy = pd.concat([X,y], axis=1)
 
# обучаем сеть (фактически сеть это вектор весов weights)
print('Стохастический')
for i in range(iterations):
    Xy = Xy.sample(frac=1)
    
    X = Xy.iloc[:, 0:2].values
    y = Xy.iloc[:, 2].values.reshape(-1,1)
      
    for k in range(len(X)):      
        net1.train(X[k], y[k])

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - net1.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net1.predict(X)
print(sum(abs(y-(pr>0.5))))

# обучаем сеть (фактически сеть это вектор весов weights)
print('не Стохастический')
for i in range(iterations):            
    net2.train(X, y)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - net2.predict(X)))))
pr = net2.predict(X)
print(sum(abs(y-(pr>0.5))))