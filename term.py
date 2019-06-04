import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy
import time

train = [
    [5,3,0,1],
    [4,0,0,1],
    [1,1,0,5],
    [1,0,0,4],
    [0,1,5,4],
]

test = [
    [0,0,2,0],
    [0,2,2,0],
    [0,0,5,0],
    [0,1,5,0],
    [1,0,0,0],
]

N = len(train)
M = len(train[0])
K = 2

P = torch.tensor(numpy.random.rand(N,K)).float().cuda()
Q = torch.tensor(numpy.random.rand(M,K)).float().cuda()
# Hyper Parameters 
input_size = 2*K
hidden_size = 2*K
output_size = 5
num_epochs = 5000
batch_size = 1
learning_rate = 0.001

# Dataset
train_dataset = torch.tensor(train)
train_dataset = train_dataset.cuda()

# test_dataset = torch.tensor(test)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, K)
        self.layer4_mlp= nn.Linear(K, output_size)
        self.layer4_mf = nn.Linear(K, output_size)
        self.layer4 = nn.Linear(K, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.P = torch.rand(N,K).cuda()
        self.Q = torch.rand(M,K).cuda()
    
    def forward(self, i,j):
        alpha = 0.5
        gmf = self.P[i] * self.Q[j]
        mlp = torch.cat((P[i],Q[j]), 0)
        out = self.layer1(mlp)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = alpha*self.sigmoid(self.layer4_mf(gmf)) + (1-alpha)*self.sigmoid(self.layer4(out))
        out = out.reshape([1,output_size])
        return out
    
net = Net(input_size, hidden_size, output_size)
net = net.cuda()
# Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
logsoftmax = nn.LogSoftmax(dim=1)


def matrix_completion(N, M, net):
    R = []
    for i in range(N):
        tmp = []
        for j in range(M):
            outputs = net(i,j)
            _,predicted = torch.max(outputs.data,1)
            tmp.append(predicted+1)
        R.append(tmp)
    return R

def transform(y):
    result = []
    for i in range(len(y)):
        tmp = []
        for j in range(len(y[0])):
            if y[i][j] == 0:
                tmp.append(0)
            else:
                tmp.append(1)
        result.append(tmp)
    return torch.tensor(result)

def one_hot_encoding(x):
    res = [0, 0, 0, 0, 0]
    if x == 5:
        res = [0, 0, 0, 0, 1]
    elif x == 4:
        res = [0, 0, 0, 1, 0]
    elif x == 3:
        res = [0, 0, 1, 0, 0]
    elif x == 2:
        res = [0, 1, 0, 0, 0]
    elif x == 1:
        res = [1, 0, 0, 0, 0]
    return res

print(train_dataset)

# y = transform(train_dataset)
# y = y.cuda()
y = train_dataset
start_time = time.time()
# Train the Model
for epoch in range(num_epochs):    
    for i in range(len(P)):
        for j in range(len(Q)):
            if y[i][j] > 0:
                # Forward + Backward + Optimize
                optimizer.zero_grad()  # zero the gradient buffer
                outputs = net(i,j)
                loss = criterion(logsoftmax(outputs.cpu()),torch.tensor([train[i][j]-1]))
                loss.backward()
                optimizer.step()
print("--- %s seconds ---" %(time.time() - start_time))
print(matrix_completion(len(P),len(Q),net))
torch.save(net.state_dict(), 'model.pkl')
