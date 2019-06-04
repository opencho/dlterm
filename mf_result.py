import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

def load_ppi_data():
    num_nodes = 10
    num_samples = 10000
    dataset_path = "/home/cho/recommender/dlterm/y2sg_n{}_s{}.npz".format(num_nodes, num_samples)
    dataset = np.load(dataset_path,allow_pickle=True)

    # Load each attributes
    G, A, F = dataset['G'], dataset['A'], dataset['F']
    
    temp_A = []
    for a in A:
        temp_A.append(a.todense())
    A = np.array(temp_A)
    
    # Debuging
    print("Sub Graphs / shape is (Samples, (Node-list, Edge-list)) : {}".format(G.shape))
    print("Adjacency matrices                                      : {}".format(A.shape))
    print("Features                                                : {}".format(F.shape))
    
    F_train, F_test, A_train, A_test  = train_test_split(F, A, test_size=0.33, random_state=42)
    print("Train A : {} / Train F :{} / Test A : {} / Test F : {}".format(A_train.shape, F_train.shape, A_test.shape, F_test.shape))
    return F_train, F_test, A_train, A_test

# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out


(F_train, F_test, A_train, A_test) = load_ppi_data()
# model = NMF(n_components=5, init='random', random_state=0)
# W = model.fit_transform(F_train[0])
# H = model.components_
# pre = np.matmul(W,H)
# pre = torch.tensor(pre).float().cuda()
F_sample = F_train
F_train = torch.tensor(F_train).float().cuda()
F_test = torch.tensor(F_test).float().cuda()
F_sample_test = F_test
# Hyper Parameters 
input_size = len(F_train[0][0])
hidden_size = len(F_train[0][0])
output_size = len(F_train[0][0])
num_epochs = 1
batch_size = 1
learning_rate = 0.001

net = torch.load('model2.pkl')
# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for i in range(len(F_sample_test)):
    for j in range(len(F_sample_test[i])):
        for k in range(len(F_sample_test[i][j])):
            if i+j+k % 3 == 0:
                F_sample_test[i][j][k] = 0

res=0
for i in range(len(F_test)):
    for j in range(len(F_test[i])):
        loss = criterion(net(F_sample_test[i][j]),F_test[i][j])
        res += loss.data
    if i % 100 == 0 and i!=0:
        print(i,"/",len(F_test), " Loss : ", res / ((i+1)*j))
print(res, res/(len(F_sample_test)*len(F_sample_test[0])))
