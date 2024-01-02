import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import docplex.mp.model as cpx


class NNMIP:
    def __init__(self, bids):
        self.bids = bids
        self.N = len(bids)
        self.M = bids[0].shape[1] - 1
        self.Mip = cpx.Model(name="WDP")
        self.K = [x.shape[0] for x in bids]
        self.z = {}
        self.x_star = np.zeros((self.N, self.M))

    def initialize_mip(self, verbose=0):
        for i in range(0, self.N):
            self.z.update({(i, k): self.Mip.binary_var(name="z({},{})".format(i, k)) for k in range(0, self.K[i])})
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, k)] for k in range(self.K[i])) <= 1), ctname="CT Allocation Bidder {}".format(i))

        for m in range(0, self.M):
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, k)]*self.bids[i][k, m] for i in range(0, self.N) for k in range(0, self.K[i])) <= 1), ctname="CT Intersection Item {}".format(m))

        objective = self.Mip.sum(self.z[(i, k)]*self.bids[i][k, self.M] for i in range(0, self.N) for k in range(0, self.K[i]))
        self.Mip.maximize(objective)

        if verbose==1:
            for m in range(0, self.Mip.number_of_constraints):
                    logging.debug('({}) %s'.format(m), self.Mip.get_constraint_by_index(m))
            logging.debug('\nMip initialized')

    def solve_mip(self, verbose=0):
        self.Mip.solve()
        if verbose==1:
            self.log_solve_details(self.Mip)
        for i in range(0, self.N):
            for k in range(0, self.K[i]):
                if self.z[(i, k)].solution_value != 0:
                    self.x_star[i, :] = self.z[(i, k)].solution_value*self.bids[i][k, :-1]

    def log_solve_details(self, solved_mip):
        details = solved_mip.get_solve_details()
        logging.debug('Status  : %s', details.status)
        logging.debug('Time    : %s sec',round(details.time))
        logging.debug('Problem : %s',details.problem_type)
        logging.debug('Rel. Gap: {} %'.format(round(details.mip_relative_gap,5)))
        logging.debug('N. Iter : %s',details.nb_iterations)
        logging.debug('Hit Lim.: %s',details.has_hit_limit())
        logging.debug('Objective Value: %s', solved_mip.objective_value)

    def summary(self):
        print('################################ OBJECTIVE ################################')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Not yet solved!\n")


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_rate):
        super(NN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 100)
        # self.fc3 = nn.Linear(150, 100)
        self.fc4 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)
        return x
    

def train(model, loader, optimizer, loss_fn):
    model.train()
    num_correct = 0
    num_samples = 0
    for x, y in loader:
        # print(x.shape)
        x = x.to(device)
        y = y.to(device)
        x = x.reshape(x.shape[0], -1)
        scores = model(x).view(BATCH_SIZE, 1)
        # print(scores.shape)
        # print(y.shape)
        loss = loss_fn(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_correct += (torch.abs(scores - y) <= 0.5).sum()
        num_samples += scores.size(0)
    return num_correct / num_samples

if __name__ == '__main__':
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    L2_STRENGTH = 0.0001
    HIDDEN_SIZE = 100
    DROPOUT_RATE = 0.2

    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NN(784, HIDDEN_SIZE, 1, DROPOUT_RATE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_STRENGTH)
    loss_fn = nn.L1Loss(model.parameters(), reduction="mean")

    # for epoch in range(NUM_EPOCHS):
    #     train_acc = train(model, train_loader, optimizer, loss_fn)
    #     print(epoch, train_acc)

    def evaluate(model, loader):
        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                x = x.reshape(x.shape[0], -1)
                scores = model(x)
                # print(scores.shape)    
                # .view(BATCH_SIZE, 1)
                num_correct += ((torch.abs(scores - y) <= 0.5).sum())/BATCH_SIZE
                # print(num_correct)
                num_samples += scores.size(0)
        return num_correct / num_samples

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    model = NN(784, HIDDEN_SIZE, 1, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("Loaded PyTorch Model State from model.pth")
    print(evaluate(model, test_loader))

    