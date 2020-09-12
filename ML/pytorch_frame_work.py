"""
Frame work for building a DCNN using Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===================================================================================
"""
Define a CNN 
"""
class Net(nn.Module):

    """
    initialize basic DCNN architecture
    """
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    """
    Construct DCNN with predefined elements 
    """

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    """
    Helper function calculate the number of flat features
    """
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# ===================================================================================

# Create DCNN object
net = Net()



# ===================================================================================
"""
Training loops
"""

# Pass in input and get output
output = net(input)

# target : correct answer for input
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

"""
Loss function
"""
# Set Loss function
criterion = nn.MSELoss()
loss = criterion(output, target)



"""
Update weights with optimizer 
"""
# create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
loss.backward()
optimizer.step()    # Does the update

# =====================================================================================
"""
Save and Load Model
"""
# Save
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Load
net = Net()
net.load_state_dict(torch.load(PATH))



# =====================================================================================
"""
Training with GPU
"""
# 1
# Define a device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2
# convert parameters to CUDA tensor
net.to(device)

# 3
# send input and target to CUDA as well
inputs, labels = data[0].to(device), data[1].to(device)
