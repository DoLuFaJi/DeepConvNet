import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from torch.optim import lr_scheduler

from dataset import FaceDataset, TestDataset, ValidDataset
from detector import Net
from run_test import run_test
from save_model import make_name
from settings import TRAIN_DATA_FACE, TRAIN_DATA_NOT_FACE, TEST_DATA_GOOGLE, \
CLASSIFIED_TEST_DATA_GOOGLE, CLASSIFIED_TEST_DATA, CLASSIFIED_TEST_DATA_YALE, \
CLASSIFIED_TEST_DATA_GOOGLE2, CLASSIFIED_TRAIN_DATA, SAVE_MODEL, MODEL_DIR, \
LOAD_MODEL, BATCH_SIZE, MOMENTUM, LEARNING_RATE, WORKERS, NB_ITERATIONS, \
SCHEDULER, MODEL_NAME, TRAIN_DATA, TEST_DATA

transform = tf.Compose([tf.RandomHorizontalFlip(), tf.RandomVerticalFlip(), tf.RandomRotation(90), tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = tf.Compose([tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = FaceDataset(TRAIN_DATA, CLASSIFIED_TRAIN_DATA, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)

testset = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

validset = FaceDataset(TRAIN_DATA, CLASSIFIED_TRAIN_DATA, transform=transform_test)
validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)


yaleset = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA_YALE, transform=transform_test)
yaleloader = torch.utils.data.DataLoader(yaleset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

googleset = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA_GOOGLE, transform=transform_test)
googleloader = torch.utils.data.DataLoader(googleset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

googleset2 = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA_GOOGLE2, transform=transform_test)
googleloader2 = torch.utils.data.DataLoader(googleset2, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

classes = ('NOT FACE', 'FACE')
if not LOAD_MODEL:
    net = Net()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # lr is learning rate
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    if SCHEDULER:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_data_size = len(trainset)
    for epoch in range(NB_ITERATIONS):
        print('Starting epoch ' + str(epoch))
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
    if SAVE_MODEL:
        torch.save(net, make_name(LEARNING_RATE, MOMENTUM, NB_ITERATIONS, BATCH_SIZE, MODEL_NAME))

trainset.close()

if LOAD_MODEL:
    net = torch.load(make_name(LEARNING_RATE, MOMENTUM, NB_ITERATIONS, BATCH_SIZE, MODEL_NAME))
net.eval()

run_test(validset, validloader, classes, net, 'VALID')
run_test(yaleset, yaleloader, classes, net, 'YALE')
run_test(googleset, googleloader, classes, net, 'GOOGLE')
run_test(googleset2, googleloader2, classes, net, 'GOOGLE2')
run_test(testset, testloader, classes, net, 'TEST')

import pdb; pdb.set_trace()
