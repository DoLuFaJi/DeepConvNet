import copy
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
from torch.optim import lr_scheduler

from dataset import FaceDataset, TestDataset, ValidDataset
from detector import Net, NetTuto
from run_test import run_test
from save_model import make_name
from show_dataset import show_dataset
from settings import TRAIN_DATA_FACE, TRAIN_DATA_NOT_FACE, TEST_DATA_GOOGLE, \
CLASSIFIED_TEST_DATA_GOOGLE, CLASSIFIED_TEST_DATA, CLASSIFIED_TEST_DATA_YALE, \
CLASSIFIED_TEST_DATA_GOOGLE2, CLASSIFIED_TRAIN_DATA, SAVE_MODEL, MODEL_DIR, \
LOAD_MODEL, BATCH_SIZE, MOMENTUM, LEARNING_RATE, WORKERS, NB_ITERATIONS, \
SCHEDULER, MODEL_NAME, TRAIN_DATA, TEST_DATA, USE_TUTO, CLASSIFIED_TEST_DATA_FACE, CLASSIFIED_TEST_DATA_OTHER,\
CLASSIFIED_TRAIN_DATA_55000, CLASSIFIED_VALID_DATA_36000

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-b', type=int)
arg_parser.add_argument('-lr', type=float)
arg_parser.add_argument('-m', type=float)
arg_parser.add_argument('-i', type=int)
arg_parser.add_argument('-n', type=str)
arg_parser.add_argument('-t', type=int)

args = arg_parser.parse_args()
if args.b is not None:
    BATCH_SIZE = args.b
if args.lr is not None:
    LEARNING_RATE = args.lr
if args.m is not None:
    MOMENTUM = args.m
if args.i is not None:
    NB_ITERATIONS = args.i
if args.n is not None:
    MODEL_NAME = args.n
if args.t is not None:
    USE_TUTO = (args.t == 1)

transform = tf.Compose([tf.RandomHorizontalFlip(), tf.RandomVerticalFlip(), tf.RandomRotation(90), tf.ColorJitter(brightness=0.5, contrast=0.75, saturation=0, hue=0), tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform = tf.Compose([tf.RandomHorizontalFlip(), tf.RandomVerticalFlip(), tf.RandomRotation(90), tf.ColorJitter(brightness=0.5, contrast=0.75, saturation=0, hue=0)])

transform_test = tf.Compose([tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = FaceDataset(TRAIN_DATA, CLASSIFIED_TRAIN_DATA_55000, transform=transform)
validset = FaceDataset(TRAIN_DATA, CLASSIFIED_VALID_DATA_36000, transform=transform)
trainloader = {'train': torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS),
                'val': torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)}

testset = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

ttset = FaceDataset(TRAIN_DATA, CLASSIFIED_TRAIN_DATA, transform=transform_test)
ttloader = torch.utils.data.DataLoader(ttset, batch_size=BATCH_SIZE,
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


# faceset = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA_FACE, transform=transform_test)
# faceloader = torch.utils.data.DataLoader(faceset, batch_size=BATCH_SIZE,
#                                          shuffle=False, num_workers=WORKERS)
#
# otherset = TestDataset(TEST_DATA, CLASSIFIED_TEST_DATA_OTHER, transform=transform_test)
# otherloader = torch.utils.data.DataLoader(otherset, batch_size=BATCH_SIZE,
#                                          shuffle=False, num_workers=WORKERS)
classes = ('NOT FACE', 'FACE')

# show_dataset(trainset)
net = None
if not LOAD_MODEL:
    net = Net()
    if USE_TUTO:
        net = NetTuto()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    if SCHEDULER:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    dataset_size = {'train': len(trainset), 'val': len(validset)}

    best_acc = 0
    best_model = copy.deepcopy(net.state_dict())
    last_best = 0
    stagnate = 0
    for epoch in range(NB_ITERATIONS):
        print('Starting epoch ' + str(epoch+1))
        for phase in ['train', 'val']:
            if phase == 'train':
                if SCHEDULER:
                    scheduler.step()
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_corrects = 0.0
            loss_epoch = 0.0
            for i, data in enumerate(trainloader[phase], 0):
                # get the inputs
                inputs, labels = data['image'], data['is_face']
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                loss_epoch += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, loss_epoch / 2000))
                    loss_epoch = 0.0

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            print('TRAIN Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    stagnate = 0
                    print('NEW best model accuracy {:.4f}'.format(epoch_acc))
                    best_acc = epoch_acc
                    last_best = epoch + 1
                    best_model = copy.deepcopy(net.state_dict())
                else:
                    stagnate += 1
                    if stagnate > 10:
                        print('Stagnated for too long, stop learning {} / {}'.format(last_best, epoch+1))


    print('Finished Training, last best was {}/{}'.format(last_best, NB_ITERATIONS))
    if SAVE_MODEL:
        torch.save(best_model, make_name(LEARNING_RATE, MOMENTUM, NB_ITERATIONS, BATCH_SIZE, MODEL_NAME))

trainset.close()

if LOAD_MODEL:
    best_model = torch.load(make_name(LEARNING_RATE, MOMENTUM, NB_ITERATIONS, BATCH_SIZE, MODEL_NAME))
print('Result for ' + make_name(LEARNING_RATE, MOMENTUM, NB_ITERATIONS, BATCH_SIZE, MODEL_NAME))
if net is not None:
    net.eval()
    # run_test(ttset, ttloader, classes, net, 'TRAINSET')
    # run_test(yaleset, yaleloader, classes, net, 'YALE')
    # run_test(googleset, googleloader, classes, net, 'GOOGLE')
    # run_test(googleset2, googleloader2, classes, net, 'GOOGLE2')
    # run_test(testset, testloader, classes, net, 'TEST')
else:
    net = Net()
    if USE_TUTO:
        net = NetTuto()
print('')
print('BEST MODEL')
print('')
net.load_state_dict(best_model)
run_test(ttset, ttloader, classes, net, 'TRAINSET')
run_test(yaleset, yaleloader, classes, net, 'YALE')
run_test(googleset, googleloader, classes, net, 'GOOGLE')
run_test(googleset2, googleloader2, classes, net, 'GOOGLE2')
run_test(testset, testloader, classes, net, 'TEST')

from detectinimage import searchfaces
searchfaces(net)
