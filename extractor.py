import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dataset import FaceDataset, TestDataset, ToTensor, ValidDataset
from detector import Net
from settings import TRAIN_DATA_FACE, TRAIN_DATA_NOT_FACE, TEST_DATA_GOOGLE, \
CLASSIFIED_TEST_DATA_GOOGLE, CLASSIFIED_TEST_DATA, CLASSIFIED_TEST_DATA_YALE, \
CLASSIFIED_TEST_DATA_GOOGLE2, CLASSIFIED_TRAIN_DATA

from run_test import run_test

BATCH_SIZE = 10
WORKERS = 2
NB_ITERATIONS = 1
SCHEDULER = False

# LEARNING_RATE = 0.000001
LEARNING_RATE = 0.001
# MOMENTUM = 0.2
# MOMENTUM = 0.5
MOMENTUM = 0.9



trainset = FaceDataset(transform=torchvision.transforms.Compose([ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)

testset = TestDataset(CLASSIFIED_TEST_DATA, transform=torchvision.transforms.Compose([ToTensor()]))
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

validset = ValidDataset(transform=torchvision.transforms.Compose([ToTensor()]))
validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)


yaleset = TestDataset(CLASSIFIED_TEST_DATA_YALE, transform=torchvision.transforms.Compose([ToTensor()]))
yaleloader = torch.utils.data.DataLoader(yaleset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

googleset = TestDataset(CLASSIFIED_TEST_DATA_GOOGLE, transform=torchvision.transforms.Compose([ToTensor()]))
googleloader = torch.utils.data.DataLoader(googleset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

googleset2 = TestDataset(CLASSIFIED_TEST_DATA_GOOGLE2, transform=torchvision.transforms.Compose([ToTensor()]))
googleloader2 = torch.utils.data.DataLoader(googleset2, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

classes = ('NOT FACE', 'FACE')

net = Net()

criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# lr is learning rate
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
if SCHEDULER:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_data_size = len(trainset)
for epoch in range(NB_ITERATIONS):
    running_loss = 0.0
    running_corrects = 0.0
    loss_epoch = 0.0
    loss_epoch_input_size = 0.0
    correct_epoch = 0.0
    if SCHEDULER:
        scheduler.step()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data['image'], data['is_face']
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs.float())
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        # import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()

        # print statistics
        loss_epoch += loss.item()
        loss_epoch_input_size += loss.item() * inputs.size(0)
        # correct_epoch += torch.sum(preds == labels.data)
        if NB_ITERATIONS > 1:
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f loss multiplied by size: %.3f' %
                  (epoch + 1, i + 1, loss_epoch / 2000, loss_epoch_input_size / 2000))
            loss_epoch = 0.0
            loss_epoch_input_size = 0.0

    if NB_ITERATIONS > 1:
        epoch_loss = running_loss / train_data_size
        epoch_acc = running_corrects.double() / train_data_size
        print('TRAIN Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

print('Finished Training')
trainset.close()
net.eval()

run_test(validset, validloader, classes, net, 'VALID')
run_test(yaleset, yaleloader, classes, net, 'YALE')
run_test(googleset, googleloader, classes, net, 'GOOGLE')
run_test(googleset2, googleloader2, classes, net, 'GOOGLE2')
run_test(testset, testloader, classes, net, 'TEST')

import pdb; pdb.set_trace()
