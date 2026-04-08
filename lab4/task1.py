import numpy as np
import pandas as pd
import torch
from torch import nn

# n = 11
# if (n % 2) == 1:
#     print('Решите задачу классификации покупателей '
#           'на классы *купит* - *не купит* (3й столбец) по признакам возраст и доход.')
#
#
# else:
#     print('Решите задачу предсказания дохода по возрасту.')

df = pd.read_csv('C:/Users/Vadim/PycharmProjects/neuro_labs/4_Создание нейронной сети на pytorch/dataset_simple.csv')
X = torch.tensor(df.iloc[:, 0:2].values, dtype=torch.float32)
y = torch.tensor(np.where(df.iloc[:, 2].values == 1, 1, -1).reshape(-1, 1), dtype=torch.float32)

# print(y)
X = (X - X.mean(dim=0)) / X.std(dim=0)
print(X)


# print(X.mean(dim=0), X.std(dim=0))
class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),  # слой линейных сумматоров
                                    nn.Tanh(),  # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Tanh()
                                    )

    # прямой проход
    def forward(self, X):
        return self.layers(X)


# задаем параметры сети
inputSize = X.shape[1]  # количество признаков задачи
hiddenSizes = 4  # число нейронов скрытого слоя
outputSize = 1  # число нейронов выходного слоя равно числу классов задачи

net = NNet(inputSize, hiddenSizes, outputSize)

lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
epochs = 1000
for i in range(epochs):
    optimizer.zero_grad()  # обнуляем градиенты
    pred = net(X)  # прямой проход - делаем предсказания
    loss = lossFn(pred, y)  # считаем ошибку
    loss.backward()
    optimizer.step()
    if (i ) % 100 == 99:
        print(f'Ошибка на {i + 1} итерации: {loss.item()}')

with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print('\nОшибка (количество несовпавших ответов): ')
print(err)
