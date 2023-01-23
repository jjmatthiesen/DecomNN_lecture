import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns


# Setting all the parameters
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.6 
momentum = 0.5
log_interval = 10

# Setting a seed and ensure CPU is used
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load MNIST dataset (that is pictures of handwritten numbers from 0-9)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# Look at some examples
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig.show()

# Define a network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# A short plotting function to save images of the gradient
def plot_wrap(i):
  sns.heatmap(network.fc2.weight.grad, cmap="Spectral", center=0)
  name = 'images/'+str(i)
  plt.savefig(name)
  plt.close()

# Initialise an instance of the network and define its optimizer
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# Initialize lists to keep track of training
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# Define training function
def train(epoch, print_grad=False):
  # Set network to training mode
  network.train()
  # Get loop to train over every batch in data
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()     # Empty the gradients
    output = network(data)    # Forward-Step // Push data through the network
    loss = F.nll_loss(output, target) # Calculate loss
    loss.backward()           # Backpropagation // Calculate and distribute gradients over network
    optimizer.step()          # Apply gradients to network weights

    # Print losses and progress
    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      epoch, batch_idx * len(data), len(train_loader.dataset),
      100. * batch_idx / len(train_loader), loss.item()))
    if print_grad:
      print(network.fc2.weight.grad[0]) # This guy selects a layer from the network and prints its gradients
    plot_wrap(batch_idx)
    train_losses.append(loss.item())
    train_counter.append(
      (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

# Evaluate model with validation data
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# Let training run
for epoch in range(1, n_epochs + 1):
  train(epoch, print_grad=False)
  test()

# Plot training loss
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
#plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig.show()
