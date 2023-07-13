import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

import torch.nn as nn
import torch.nn.functional as F

nlayers = 4
nfilters = 96
lambda_s = 100
lambda_v = 10
tau = 0.5
lr = 0.003
max_epochs = 1000
condnet_min_prob = 0.1
condnet_max_prob = 0.75

class policyNet(nn.Module):
    def __init__(self, n_layer=4, hidden_s = [1024, 1024, np.nan]):
        super().__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(nn.Linear(32*32*3, 32*32*3))



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        # flatten
        x = x.view(-1, 32*32*3)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # softmax
        x = F.softmax(x, dim=1)
        return x
def main():

    BATCH_SIZE = 1024
    train_dataset = datasets.CIFAR10(
        root="../data/cifar10",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root="../data/cifar10",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # create model
    model = Net()
    learning_rate = 0.05

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(train_loader, 0):
            bn += 1
            # get batch
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += loss.item()
            accs += acc
            print('Epoch: {}, Batch: {}, Cost: {:.35}, Acc: {:.3f}'.format(epoch, i, loss.item(), acc))


        scheduler.step()

        # print epoch and epochs costs and accs
        print('Epoch: {}, Cost: {}, Accuracy: {}'.format(epoch, costs / bn, accs / bn))

        # calculate accuracy on test set
        acc = 0
        bn = 0
        for i, data in enumerate(test_loader, 0):
            bn += 1
            # get batch
            inputs, labels = data

            # make one hot vector
            y_batch_one_hot = torch.zeros(labels.shape[0], 10)
            y_batch_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1,).tolist()] = 1

            # get output
            output = model(torch.tensor(inputs))

            # calculate accuracy
            pred = torch.argmax(output, dim=1)
            acc += torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
        #print accuracyt
        print('Test Accuracy: {}'.format(acc / bn))



if __name__=='__main__':
    main()