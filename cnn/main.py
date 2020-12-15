import pylab as plt
import neuralnetwork

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

#Change variables and path here
trainingfile = ['trainset.csv']
testingfile = ['test.csv']

epoch = 100
device = T.device('cpu')

#Loading the dataset
train_dataset = utils.FenDataset(trainingfile)
test_dataset = utils.FenDataset(testingfile)

print('Samples:',len(dataset), 'Total,', len(train_dataset),'Train,', len(test_dataset), 'Test.')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# NN
model = neuralnetwork.CNN()
model.to(device)
optimizer = T.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def train_step(x, t, nn, opt, loss):
    y = nn(x)
    loss = loss(y, t)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

acc_hist_train = []
acc_hist_test = []
loss_hist_train = []
loss_hist_test = []
for epoch in range(epoch):
    # Training
    model.train()
    pbar = tqdm(total=len(train_loader))
    train_loss = 0
    correct=0
    pbar.set_description('Training ('+str(epoch)+')')
    for i, (data, target, fen) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_step(data, target, model, optimizer, loss_fn)
        output = model(data)
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        if i % 810 == 0:
            pbar.set_postfix(loss=loss.item())
        pbar.update(1)
    pbar.close()

    train_loss /= len(train_loader.dataset)
    loss_hist_train.append(train_loss)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    print(train_accuracy)
    acc_hist_train.append(train_accuracy)

    # Testing
    model.eval()

    test_loss = 0
    correct = 0

    for data, target, fen in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    acc_hist_test.append(accuracy)
    loss_hist_test.append(test_loss)
    print('\tTest set: Average loss: {}, Accuracy: {}/{} ({}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

T.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'TheModel.pth')

print(acc_hist_train)
print(acc_hist_test)
print(loss_hist_train)
print(loss_hist_test)


plt.plot(range(len(acc_hist_train)), acc_hist_train)
plt.plot(range(len(acc_hist_test)), acc_hist_test)
plt.show()