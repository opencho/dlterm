import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy
import time

train = [
    [5.0,3,0,1],
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

P = torch.tensor(numpy.random.rand(N,K))
Q = torch.tensor(numpy.random.rand(M,K))

# Hyper Parameters 
input_size = 2*K
hidden_size = 2*K
last_hidden_size = 3*K
output_size = 1
num_epochs = 500
batch_size = 1
learning_rate = 0.001

# Dataset
train_dataset = torch.tensor(train)

# test_dataset = torch.tensor(test)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(last_hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        GMF = x[0 : 2] * x[2 : 4]
        MLP = x.view(x.numel()).float()
        out = self.layer1(MLP)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        out = torch.cat((GMF.float(),out), 0).float()
        # out = out.view(out.numel())
        out = self.layer5(out)
        return out
    
# Loss and Optimizer
criterion = nn.MSELoss()  

def matrix_factorization(Y, P, Q, K, steps=500, alpha=0.0002, beta=0.02):
    Y = Y.cuda()
    for step in range(steps):
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] > 0:
                    net = Net(input_size, hidden_size, output_size)
                    net.cuda()
                    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
                    # Train the Model
                    for epoch in range(num_epochs):
                        # Forward + Backward + Optimize
                        optimizer.zero_grad()  # zero the gradient buffer
                        inputs = torch.cat((P[i],Q[j]), 0).cuda()
                        outputs = net(inputs)
                        loss = criterion(outputs, Y[i][j])
                        loss.backward()
                        optimizer.step()
                        P[i] = P[i] + alpha * (2 * loss.data * Q[j] - beta * P[i])
                        Q[j] = Q[j] + alpha * (2 * loss.data * P[i] - beta * Q[j])
    return P, Q

start_time = time.time() 
nP, nQ = matrix_factorization(train_dataset, P, Q, K)
print("--- %s seconds ---" %(time.time() - start_time))

print(nP.mm(nQ.t()))

# torch.save(net.state_dict(), 'model.pkl')