import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# %%
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(
    root='./lab5/data1/train', transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(
    root='./lab5/data1/test', transform=data_transforms)

# print(train_dataset.classes)
class_names = train_dataset.classes
# %%
batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=2)

# net = torchvision.models.alexnet(weights=True)


net = torchvision.models.alexnet(weights=None)

state_dict = torch.load('./lab5/alexnet-owt-7be5be79.pth', map_location=device)
net.load_state_dict(state_dict)

for param in net.parameters():
    param.requires_grad = False

num_classes = len(class_names)

new_classifier = net.classifier[:-1]
new_classifier.add_module('fc', nn.Linear(4096, num_classes))
net.classifier = new_classifier

net = net.to(device)
# %%

num_epochs = 5
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

save_loss = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = lossFn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_loss.append(loss.item())
        if i % 100 == 0:
            print(f'Эпоха {epoch + 1} из {num_epochs} Ошибка: {loss.item()}')

plt.figure()
plt.plot(save_loss)

correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print(f'Точность модели: {round(100 * correct_predictions / num_test_samples, 3)}%')

inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
pred_class = torch.max(pred.data, 1)[1]

for i, j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(1)
