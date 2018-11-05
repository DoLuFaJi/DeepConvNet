import torch
import datetime
import pprint

from settings import SAVE_NOT_DETECTED

def run_test(set, loader, classes, net, test_label):
    correct = 0
    total = 0
    # accruacy of each classes
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    not_detected = list([] for i in range(2))
    with torch.no_grad():
        for data in loader:
            images, labels, img_names = data['image'], data['is_face'], data['image_name']
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            result = (predicted == labels).int()
            correct += result.sum().item()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += result[i].item()
                class_total[label] += 1
                if result[i].item() != 1:
                    not_detected[label].append(img_names[i])

    print('%s: Accuracy of the network: %d %d %% [%d / %d]' % (
        test_label, len(set), 100 * correct / total, correct, total))

    for i in range(2):
        if class_total[i] > 0:
            percentage = 100 * class_correct[i] / class_total[i]
        else:
            percentage = 100 * class_correct[i] / 1

        print('%s: %5s : %2d %% [%d / %d]' % (
            test_label, classes[i], percentage, class_correct[i], class_total[i]))

    set.close()
    if SAVE_NOT_DETECTED:
        filename = test_label + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(filename, 'w') as file:
            file.write(str(not_detected))
        
