import torch
import pickle
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets # 데이터를 다루기 위한 TorchVision 내의 Transforms와 datasets를 따로 임포트

import torch.nn as nn
import torch.nn.functional as F

nlayers = 2
nfilters = 96
lambda_s = 100
lambda_v = 10
tau = 0.5
lr = 0.003
max_epochs = 1000
condnet_min_prob = 0.5
condnet_max_prob = 1

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


class model_condnet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 32*32*3
        mlp_hidden = 1024
        output_dim = 10

        self.mlp_nlayer = 0

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_dim, mlp_hidden))
        for i in range(nlayers):
            self.mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
        self.mlp.append(nn.Linear(mlp_hidden, output_dim))

        n_each_policylayer = 1
        # n_each_policylayer = 1 # if you have only 1 layer perceptron for policy net
        self.policy_net = nn.ModuleList()
        temp = nn.ModuleList()
        temp.append(nn.Linear(input_dim, mlp_hidden))
        temp.append(nn.Linear(mlp_hidden, mlp_hidden))
        self.policy_net.append(temp)

        for i in range(len(self.mlp)-2):
            temp = nn.ModuleList()
            for j in range(n_each_policylayer):
                temp.append(nn.Linear(self.mlp[i].out_features, self.mlp[i].out_features))
            self.policy_net.append(temp)
    def forward(self, x):
        # return policies
        policies = []
        sample_probs = []
        layer_masks = []

        x = x.view(-1, 32*32*3)

        # for each layer
        h = x
        u = torch.ones(h.shape[0], h.shape[1])

        for i in range(len(self.mlp)-1):
            # h_clone = h.clone()
            # p_i = self.policy_net[i][0](h_clone.detach())
            p_i = self.policy_net[i][0](h)

            p_i = F.sigmoid(p_i)
            for j in range(1, len(self.policy_net[i])):
                p_i = self.policy_net[i][j](p_i)
                p_i = F.sigmoid(p_i)

            p_i = p_i * (condnet_max_prob - condnet_min_prob) + condnet_min_prob
            u_i = torch.bernoulli(p_i)

            # debug[TODO]
            u_i = torch.ones(u_i.shape[0], u_i.shape[1])

            if u_i.sum() == 0:
                idx = np.random.uniform(0, u_i.shape[0], size = (1)).astype(np.int16)
                u_i[idx] = 1

            sampling_prob = p_i * u_i + (1-p_i) * (1-u_i)

            # idx = torch.where(u_i == 0)[0]

            # h_next = F.relu(self.mlp[i](h*u.detach()))*u_i
            h_next = F.relu(self.mlp[i](h*u))*u_i
            h = h_next
            u = u_i

            policies.append(p_i)
            sample_probs.append(sampling_prob)
            layer_masks.append(u_i)

        # last layer just go without dynamic sampling
        h = self.mlp[-1](h)
        h = F.softmax(h, dim=1)
        return h, policies, sample_probs, layer_masks

def main():
    # main for the condnet

    BATCH_SIZE = 1000
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
    model = model_condnet()
    # model = model_condnet2()
    learning_rate = 0.1


    # lambda_s, lambda_v, lambda_l2
    lambda_s = 1e-1 #100
    lambda_v = 1e-3 #10
    tau = 0.8
    # lambda_l2 = 5e-4
    lambda_l2 = 0
    lambda_pg = 1e-5

    C = nn.CrossEntropyLoss()
    mlp_optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=lambda_l2)
    # mlp_optimizer = optim.SGD(model.mlp.parameters(), lr=learning_rate,
    #                       momentum=0.9, weight_decay=lambda_l2)
    policy_optimizer = optim.SGD(model.policy_net.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=lambda_l2)

    # run for 50 epochs
    for epoch in range(max_epochs):

        model.train()
        costs = 0
        accs = 0

        bn = 0
        # run for each batch
        for i, data in enumerate(train_loader, 0):
            mlp_optimizer.zero_grad()
            policy_optimizer.zero_grad()

            bn += 1
            # get batch
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고

            # 순전파 + 역전파 + 최적화를 한 후
            outputs, policies, sample_probs, layer_masks  = model(inputs)

            # make labels one hot vector
            y_one_hot = torch.zeros(labels.shape[0], 10)
            y_one_hot[torch.arange(labels.shape[0]), labels.reshape(-1)] = 1

            c = C(outputs, labels)

            # L =  c + lambda_s * \
            #      torch.add( (((torch.stack(policies)).mean(axis=1)-tau)**2).mean(), \
            #        (((torch.stack(policies)).mean(axis=2)-tau)**2).mean()) + \
            #          lambda_v * (-1) * torch.add(((torch.stack(sample_probs)).var(axis=1)).mean(),\
            #                         ((torch.stack(sample_probs)).var(axis=2)).mean())
            #     # batch L_b and

            # Compute the regularization loss L

            L = c + lambda_s * (torch.pow(torch.stack(policies).mean(axis=1) - tau, 2).mean() +
                                torch.pow(torch.stack(policies).mean(axis=2) - tau, 2).mean() +
                                lambda_v * (-1) * (torch.stack(sample_probs).var(axis=1).mean() +
                                                   torch.stack(sample_probs).var(axis=2).mean()))



            # Compute the policy gradient (PG) loss
            logp = torch.log(torch.cat(sample_probs)).sum(axis=1).mean()
            PG = lambda_pg * c * (-logp) + L

            # logp = torch.log(torch.concatenate(sample_probs)).sum(axis=1).mean()
            # PG = c * (-logp) + L
            PG.backward() # it needs to be checked [TODO]
            # c.backward()
            mlp_optimizer.step()
            # policy_optimizer.step()

            # calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            acc = torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]

            # addup loss and acc
            costs += c.item()
            accs += acc

            # print PG.item(), and acc with name
            print('Epoch: {}, Batch: {}, Cost: {:.35}, PG:{:.3f}, Acc: {:.3f}, Tau: {:.2f}'.format(epoch, i, c.item(), PG.item(), acc, torch.stack(layer_masks).mean().item()))

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
            output,_,_,_ = model(torch.tensor(inputs))

            # calculate accuracy
            pred = torch.argmax(output, dim=1)
            acc += torch.sum(pred == torch.tensor(labels.reshape(-1))).item() / labels.shape[0]
        #print accuracyt
        print('Test Accuracy: {}'.format(acc / bn))



def main_():

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
    learning_rate = 0.01

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