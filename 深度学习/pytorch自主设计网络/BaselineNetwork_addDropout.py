# -- coding: utf-8 --
# -- coding: utf-8 --
import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv4 = nn.Conv2d(32, 96, 3, padding=1)
        self.conv5 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv6 = nn.Conv2d(96, 96, 3, padding=1)

        self.conv7 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv9 = nn.Conv2d(192, 192, 3)

        self.maxpool = nn.MaxPool2d(2, 2)
        # self.avgpool = nn.AvgPool2d(2, 2)
        # self.globalavgpool = nn.AvgPool2d(3, 3)

        self.fc1 = nn.Linear(192 * 3 * 3, 10)

    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)

        # 第二个卷积块
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool(x)
        x = nn.Dropout(0.2)(x)

        # 第三个卷积块
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.maxpool(x)
        x = nn.Dropout(0.5)(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def train_sgd(self, device):
        optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.01)
        expLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        loss = nn.CrossEntropyLoss()
        for epoch in range(45):  # loop over the dataset multiple times
            timestart = time.time()
            running_loss = 0.0
            total = 0
            correct = 0
            expLR.step()
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                path = 'model.th'

                # print statistics
                running_loss += l.item()
                if i % 500 == 499:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, (i+1)*100, running_loss / 100))
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d train iterations: %.3f %%' % (total*5, 100.0 * correct / total))

                    torch.save({'epoch': epoch,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                }, path)

            print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))

        print('Finished Training')

    def test(self, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    net.train_sgd(device)
    net.test(device)
