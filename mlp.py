import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

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

# train = [
#     [5.0,3,2,1],
#     [4,2,2,1],
#     [1,1,5,5],
#     [1,1,5,4],
#     [1,1,5,4]
# ]


# Hyper Parameters 
input_size = 4
hidden_size = 4
ouput_size = 4
num_epochs = 5000
batch_size = 1
learning_rate = 0.001

# MNIST Dataset 
train_dataset = torch.tensor(train)

test_dataset = torch.tensor(test)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, ouput_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        return out
    
net = Net(input_size, hidden_size, ouput_size)
net.cuda()
    
# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# Train the Model
for epoch in range(num_epochs):
    inputs = train_dataset.cuda()

    # Forward + Backward + Optimize
    optimizer.zero_grad()  # zero the gradient buffer
    outputs = net(inputs)
    loss = criterion(outputs, inputs)
    loss.backward()
    optimizer.step()
    print ('Epoch [%d/%d],  Loss: %.4f' 
                   %(epoch+1, num_epochs, loss.data))
    print(outputs.data)

# Test the Model
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images.view(-1, 28*28)).cuda()
#     outputs = net(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()

# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')