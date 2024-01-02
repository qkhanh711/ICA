import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm

transforms = transforms.Compose([
    transforms.ToTensor()
])

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
L1_STRENGTH = 0.01
L2_STRENGTH = 0.01

train_dataset = datasets.MNIST(root='data/', train=True,
                            transform=transforms, download=True)
test_dataset = datasets.MNIST(root='data/', train=False,
                            transform=transforms, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)



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
        return x.view(BATCH_SIZE, 1)    

model = NN(784, 10, 1, 0.2).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_STRENGTH)
loss_fn = nn.L1Loss(model.parameters(), reduction="mean")

for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        # print(data.shape)
        data = data.to(device)
        targets = (targets).to(device)
        
        data = data.reshape(data.shape[0], -1)
        
        scores = model(data)
        # print(scores.shape)
        # print(targets.shape)
        loss = loss_fn(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    print(epoch,loss)

def plot_results(model, loader):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            predictions = torch.argmax(scores, dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()

plot_results(model, train_loader)
plot_results(model, test_loader)

if __name__ == "__main__":

    print("NN class is being run directly!")
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(data.shape)
        print(targets.shape)
        break