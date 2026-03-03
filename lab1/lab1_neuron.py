import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('lab1/data3.csv')
shuffled_df = df.sample(frac=1)

print(shuffled_df.head())

#%%
y = shuffled_df.iloc[:, 4].values
y = np.where(y != "Iris-setosa", y, 1)
y = np.where(y != "Iris-versicolor", y, -1)
y = np.where(y != "Iris-virginica", y, 0)

X = shuffled_df.iloc[:, 0:3].values

def neuron(w, x):
    if (w[1]*x[0] + w[2]*x[1] + w[3]*x[2] + w[0]) >= 0:
        predict = 1
    else:
        predict = -1
    return predict

#%%
w1 = np.random.random(4)
w2 = np.random.random(4)
w3 = np.random.random(4)
max_acc = 0
epoch = 0

eta = 0.01

#%%
w_iter = []


for xi, target, j in zip(X, y, range(X.shape[0])):
    if target != -1:
        predict1 = neuron(w1, xi)
        if target == 1:
            w1[1:] += (eta * (target - predict1)) * xi
            w1[0] += eta * (target - predict1)
        else:
            w1[1:] += (eta * (target - predict1 - 1)) * xi
            w1[0] += eta * (target - predict1 - 1)
    
    if target != 0:
        predict2 = neuron(w2, xi)
        w2[1:] += (eta * (target - predict2)) * xi
        w2[0] += eta * (target - predict2)
    
    if target != 1:
        predict3 = neuron(w3, xi)
        if target == -1:
            w3[1:] += (eta * (target - predict3)) * xi
            w3[0] += eta * (target - predict3)
        else:
            w3[1:] += (eta * (target - predict3 + 1)) * xi
            w3[0] += eta * (target - predict3 + 1)

sum_err = 0
for xi, target in zip(X, y):
    if neuron(w1, xi) == 1 and neuron(w2, xi) == 1:
        predict = 1
    elif neuron(w1, xi) == -1 and neuron(w3, xi) == 1:
        predict = 0
    elif neuron(w2, xi) == -1 and neuron(w3, xi) == -1:
        predict = -1
    else:
        predict = neuron(w1, xi)
    
    if target != predict:
        sum_err += 1

sum_err = abs(sum_err)
acc = 100 - sum_err / 150 * 100
max_acc = acc if acc > max_acc else max_acc
epoch += 1
print(f"{epoch}\nВсего ошибок: {sum_err:3} | Точность: {acc:3.2f} %\nМаксимальная точность: {max_acc:3.2f} %")

