import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from dataset import FaceDataset, TestDataset, ToTensor
from detector import Net
from settings import TRAIN_DATA_FACE, TRAIN_DATA_NOT_FACE, TEST_DATA_GOOGLE

BATCH_SIZE = 4
WORKERS = 2
NB_ITERATIONS = 50

trainset = FaceDataset(transform=torchvision.transforms.Compose([ToTensor()]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)

testset = TestDataset(transform=torchvision.transforms.Compose([ToTensor()]))

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)

validset = FaceDataset(transform=torchvision.transforms.Compose([ToTensor()]))

validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)


classes = ('NOT FACE', 'FACE')

net = Net()

criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# lr is learning rate
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
train_data_size = len(trainset)
for epoch in range(NB_ITERATIONS):
    running_loss = 0.0
    running_corrects = 0.0
    loss_epoch = 0.0
    loss_epoch_input_size = 0.0
    correct_epoch = 0.0
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

# Testing 1 image
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# outputs = net(images)
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# import pdb; pdb.set_trace()

net.eval()
### VALID SET, same set as train set
correct = 0
total = 0
# accruacy of each classes
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in validloader:
        images, labels = data['image'], data['is_face']
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        result = (predicted == labels).int()
        correct += result.sum().item()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += result[i].item()
            class_total[label] += 1


print('Accuracy of the network on the %d valid images: %d %% [%d / %d]' % (
    len(validset), 100 * correct / total, correct, total) )

for i in range(2):
    print('Accuracy of %5s : %2d %% [%d / %d]' % (
        classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
validset.close()
print('Starting test')
### TEST SET, google x2, yale
correct = 0
total = 0
# accruacy of each classes
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in testloader:
        images, labels = data['image'], data['is_face']
        outputs = net(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        result = (predicted == labels).int()
        correct += result.sum().item()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += result[i].item()
            class_total[label] += 1


print('Accuracy of the network on the %d test images: %d %% [%d / %d]' % (
    len(testset),100 * correct / total, correct, total) )

for i in range(2):
    print('Accuracy of %5s : %2d %% [%d / %d]' % (
        classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
testset.close()
import pdb; pdb.set_trace()
