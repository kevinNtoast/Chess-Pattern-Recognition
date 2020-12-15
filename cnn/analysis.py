import chess.svg
import numpy as np
import neuralnetwork
import torch as T
from torch.utils.data import DataLoader
import torch.functional as F
import utils

#Initializing the neural network
model = neuralnetwork.CNN()
optimizer = T.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

checkpoint = T.load('TheModel.pth', map_location=T.device('cpu'))

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

dataset = utils.FenDataset(['trainset.csv'])
a,b, line = dataset[1]
fen, best_move = line.split(',')
board = chess.Board(fen)

col = int(ord(best_move[0])-96)
row = int(best_move[1])
#
piece = a.numpy()[1:13, row-1, col-1]
itemindex = np.where(piece==1)


display.start(board.fen())
while not display.checkForQuit():
    sleep(100000)
display.terminate()

dataset = utils.FenDataset(['testset.csv'])
test = DataLoader(dataset)
# T.set_printoptions(edgeitems = 100)
fenwrong = []
fenright = []
file = open('analysis3.csv', 'w')

def t2s(tup):
    str =  ''.join(tup)
    return str

test_loss = 0
correct = 0

data,target,fen = test[1]
output = model(data)
out = T.argmax(output, dim=1)

test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
right = pred.eq(target.view_as(pred)).sum().item()
correct = correct + right

print(right)
print(target.view_as(pred).sum.item())

ind = 0
for data, target, fen in test:
    output = model(data)
    out = T.argmax(output, dim=1)

    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    right = pred.eq(target.view_as(pred)).sum().item()
    correct = correct + right

    if right == 0:
        file.write(t2s(fen))
        file.write('\n')
        file.flush()

    fen, best_move = line.split(',')
    board = chess.Board(fen)

    col = int(ord(best_move[0]) - 96)
    row = int(best_move[1])

    piece = data.numpy()[1:13, row-1, col-1]
    itemindex = np.where(piece==1)

    ind += 1
    print(ind)

accuracy = 100. * correct / len(test.dataset)

print("Accuracy:")
print(accuracy)

print(fenright)
print(fenwrong)

wrongfen = utils.FenDataset(['testset.csv'])
a,b, line = wrongfen[1]
fen, best_move = line.split(',')
board = chess.Board(fen)

col = int(ord(best_move[0])-96)
row = int(best_move[1])
#
#
# piece = a.numpy()[1:13, row-1, col-1]
# itemindex = np.where(piece==1)

total = np.zeros(6)


for a, b, line in wrongfen:
    fen, best_move = line.split(',')
    col = int(ord(best_move[0]) - 96)
    row = int(best_move[1])

    piece = a.numpy()[1:13, row-1, col-1]
    index = np.where(piece == 1)

    total[index] += 1

print(total)