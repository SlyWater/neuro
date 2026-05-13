# -*- coding: utf-8 -*-
"""
Created for lab7 from Lab6_Bert_nlp_ru_toxic.py style.

BERT Fine-Tuning Sentence Classification

В этой версии используется свой датасет my_dataset_3classes.csv:
0 - технологии
1 - спорт
2 - кулинария
"""

import torch
import numpy as np
import os
import pandas as pd


# Проверяем доступна ли GPU и задаем вычислительное устройство
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Available GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU')
    device = torch.device("cpu")


# В работе используется свой набор текстов на 3 класса:
# 0 - технологии
# 1 - спорт
# 2 - кулинария
#
# Файл датасета должен лежать в lab7/data/my_dataset_3classes.csv
# и иметь две колонки: sentence,label

# Загрузка данных реализована на основе pandas dataframe
df = pd.read_csv("lab7/data/my_dataset_3classes_big.csv", delimiter=',', header=0)

print('В наборе предложений: {:,}\n'.format(df.shape[0]))

# Пример
df.sample(10)

# Нас интересуют метки классов и сами предложения, на них мы будем обучать нашу сеть
sentences = df.sentence.values
labels = df.label.values

# Следующий этап - токенизация - разбиение предложений на слова
from transformers import BertTokenizer

# Используем BERT tokenizer, но созданный на основе словаря из лабораторной
tokenizer = BertTokenizer('lab7/data/vocab_rutoxic.txt', do_lower_case=True, do_basic_tokenize=True, never_split=None)
#%%
##### размер нового словаря
tokenizer.vocab_size

# максимальный размер предложения существенно вырос
sl = [len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))) for sen in sentences]
print('Максимальная длина предложения: ', max([len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen))) for sen in sentences]))

# посмотрим сколько предложений имеет длину более 64 токена
value_c = pd.Series(sl).value_counts()
print('Предложений длиннее 64 токена: ', sum(value_c[64:]))

# довольно много, но придется ими пожертвовать
# будем их обрезать

input_ids = np.zeros((len(sentences), 64))

# Каждое предложение энкодится по отдельности
for s, i in zip(sentences, range(len(sentences))):
    enc_s = tokenizer.encode(s,
                        add_special_tokens=True, # Указываем, что нам нужно добавить служебные токены
                        padding='max_length',    # дополнение до макс.длины
                        max_length=64,           # максимальная длина предложений
                        truncation=True          # все что длиннее max_length будет обрезаться
                   )
    # Формируем список id токенов
    input_ids[i,] = enc_s


# Создаем attention mask для виртуальных токенов
attention_masks = []

for s in input_ids:
    #   Если ID = 0, это виртуальный токен и маска для него 0.
    #   Если ID > 0, это реальный токен и маска для него 1.
    att_mask = [int(id_ > 0) for id_ in s]
    attention_masks.append(att_mask)


# Формируем тестовый и валидационный набор
from sklearn.model_selection import train_test_split

# разбиваем данные, метки классов
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels,
)
# и маску
train_masks, validation_masks, _, _ = train_test_split(
    attention_masks,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels,
)

# все конвертируем в тензоры
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# теперь можно создавать Dataset и DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# размер батча придется уменьшить (если на GPU), т.к. за счет
# увеличившегося словаря выросла и модель
batch_size = 4

# Пакуем в тренировочный предложения (ID), маску и метки классов
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# и в валидационный
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#%%
# теперь можно переходить к заданию модели

from transformers import BertForSequenceClassification, BertConfig

# Загрузка теперь делается через конфигурационный файл, в котором изменен размер словаря
configuration = BertConfig.from_pretrained('lab7/data/config_rutoxic.json')

configuration.num_labels = 3
configuration.id2label = {
    0: "технологии",
    1: "спорт",
    2: "кулинария",
}
configuration.label2id = {
    "технологии": 0,
    "спорт": 1,
    "кулинария": 2,
}

# Для совместимости с разными версиями transformers.
configuration.classifier_dropout = None

model = BertForSequenceClassification(configuration)

#%%
# Отправляем модель на GPU
if torch.cuda.is_available():
    model.cuda()

# Задаем оптимизатор
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=2e-5, # скорость обучения
                              eps=1e-8 # специфический параметр, повышающий стабильность обучения
                            )
#%%
from transformers import get_linear_schedule_with_warmup

# Количество эпох обучения
epochs = 4

# Шагов обучения = number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# scheduler - планировщик изменяющий скорость обучения
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# функция вычисления точности обучения
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#%%
import time
import datetime
import random

# Задаем seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Тут храним наши лоссы
loss_values = []

# Цикл обучения будет состоять из обучения и валидации
for epoch_i in range(0, epochs):

    ################ Часть обучения #####################

    print("")
    print('Эпоха {:} из {:} '.format(epoch_i + 1, epochs))

    t0 = time.time()

    # потери за эпоху
    total_loss = 0

    # переключаем в режим обучения
    model.train()

    # пробегаем по батчам
    for step, batch in enumerate(train_dataloader):

        # Достаем из батча данные: предложения, маску и метки
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # обнуляем градиенты
        model.zero_grad()

        # прямой проход
        outputs = model(b_input_ids.to(torch.long),
                    token_type_ids=None,
                    attention_mask=b_input_mask.to(torch.long),
                    labels=b_labels.to(torch.long))

        # потери
        loss = outputs.loss
        total_loss += loss.item()

        # обратный проход
        loss.backward()

        # обрезаем градиенты до 1.0, чтобы предотвратить "взрыв градиентов"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # обновляем параметры модели
        optimizer.step()

        # изменяем скорость обучения
        scheduler.step()

        # диагностическую информацию выводим каждые 10 батчей
        if step % 10 == 0 and not step == 0:
            time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
            print(' Батч {:>4,} из {:>4,}. Затраченное время: {:}. Ошибка: {:}.'.format(step, len(train_dataloader), time_elapsed, loss))

    # средний loss
    avg_train_loss = total_loss / len(train_dataloader)

    # сохраним для графика
    loss_values.append(avg_train_loss)

    print("")
    print(" Средний loss: {0:.2f}".format(avg_train_loss))
    print(" Обучение эпохи прошло за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))

    ################ Часть валидации #####################
    # Позволяет понять правильно ли мы учимся и учимся ли вообще

    print("\n Validation...")
    t0 = time.time()

    # модель в evaluation режим
    model.eval()

    # диагностические переменные
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # пробегаем валидационный набор
    for batch in validation_dataloader:

        batch = tuple(t.to(device) for t in batch)

        # берем нужные данные из батча
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # прямой проход
            outputs = model(b_input_ids.to(torch.long),
                    token_type_ids=None,
                    attention_mask=b_input_mask.to(torch.long),
                    labels=b_labels.to(torch.long))

        # "logits" хранят вероятности классов похоже на softmax
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # считаем точность модели на валидации
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # суммарная точность
        eval_accuracy += tmp_eval_accuracy

        # сколько батчей прошло
        nb_eval_steps += 1

    # результат валидации
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Валидация прошла за: {:}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))

#%%

# Можно построить график обучения
import matplotlib.pyplot as plt

plt.plot(loss_values, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


#%%
# Сохраним обученную модель и tokenizer, чтобы потом можно было быстро загрузить
model_dir = 'lab7/model_save_my_3classes/'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)


# Загрузим обученную модель
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

model.to(device)


# Проверим как оно работает
classes = {
    0: "технологии",
    1: "спорт",
    2: "кулинария",
}

model.eval()

test_sentences = [
    'Новый процессор работает быстрее и потребляет меньше энергии.',
    'Футболист забил решающий гол на последней минуте матча.',
    'Добавьте в тесто муку, яйца и немного сахара.',
    'Разработчики выпустили обновление операционной системы.',
    'Команда выиграла турнир после серии пенальти.',
    'Суп нужно варить на медленном огне около двадцати минут.',
]

for sentence in test_sentences:
    enc_s = tokenizer.encode(sentence,
                            add_special_tokens=True, # Указываем, что нам нужно добавить служебные токены
                            padding='max_length',    # дополнение до макс.длины
                            max_length=64,           # максимальная длина предложений
                            truncation=True          # все что длиннее max_length будет обрезаться
                       )

    # Формируем список id токенов
    input_ids = np.array(enc_s)

    # Создаем attention mask для виртуальных токенов
    attention_mask = [int(id_ > 0) for id_ in input_ids]

    batch = tuple(t.to(device) for t in torch.Tensor([input_ids, attention_mask]))
    b_input_ids, b_input_mask = batch
    with torch.no_grad():
        outputs = model(b_input_ids.unsqueeze(0).to(torch.long), token_type_ids=None, attention_mask=b_input_mask.unsqueeze(0))

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    predicted_label = np.argmax(logits, axis=1).flatten()
    predicted_class_id = int(predicted_label[0])

    print('Предложение:', sentence)
    print('Предсказанный класс:', predicted_class_id, '-', classes[predicted_class_id])
    print('-' * 80)

#%%
# Отдельный блок для запуска уже обученной модели из файла.
# Его можно запускать отдельно, без повторного обучения модели выше.
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = './lab7/model_save_my_3classes/'

loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
loaded_tokenizer = BertTokenizer.from_pretrained(model_dir)

loaded_model.to(device)
loaded_model.eval()

classes = {
    0: "технологии",
    1: "спорт",
    2: "кулинария",
}

test_sentences = [
    'Новый процессор работает быстрее и потребляет меньше энергии.',
    'Футболист забил решающий гол на последней минуте матча.',
    'Добавьте в тесто муку, яйца и немного сахара.',
    'Разработчики выпустили обновление операционной системы.',
    'Команда выиграла турнир после серии пенальти.',
    'Суп нужно варить на медленном огне около двадцати минут.',
]

for sentence in test_sentences:
    enc_s = loaded_tokenizer.encode(sentence,
                            add_special_tokens=True, # Указываем, что нам нужно добавить служебные токены
                            padding='max_length',    # дополнение до макс.длины
                            max_length=64,           # максимальная длина предложений
                            truncation=True          # все что длиннее max_length будет обрезаться
                       )

    input_ids = np.array(enc_s)
    attention_mask = [int(id_ > 0) for id_ in input_ids]

    b_input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    b_input_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = loaded_model(
            b_input_ids.to(torch.long),
            token_type_ids=None,
            attention_mask=b_input_mask.to(torch.long),
        )

    logits = outputs.logits.detach().cpu().numpy()
    predicted_class_id = int(np.argmax(logits, axis=1)[0])

    print('Предложение:', sentence)
    print('Предсказанный класс:', predicted_class_id, '-', classes[predicted_class_id])
    print('-' * 80)
