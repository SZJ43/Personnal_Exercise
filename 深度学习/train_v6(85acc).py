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

# print(len(testset))
# input()
# print("trainset", trainset)
# print("testset:", testset)
# print("len(trainloder):", len(trainloader))
# print("len(testloader):", len(testloader))
# input()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  # 10*10
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(6, 6)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.dropout50 = nn.Dropout(0.5)  # 这里的参数是丢弃率，不是保留率
        self.dropout10 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(256 * 6 * 6, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout10(x)

        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.bn3(F.relu(self.conv7(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout50(x)
        x = self.fc1(x)

        return x

    def train_sgd(self, device):
        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
        loss = nn.CrossEntropyLoss()
        for epoch in range(50):  # loop over the dataset multiple times
            timestart = time.time()
            running_loss = 0.0
            total = 0
            correct = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                # print(outputs.shape)
                # print(labels.shape)
                # input()
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                path = 'model.th'

                # print statistics
                running_loss += l.item()
                # print("i ", i)
                if i == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i, running_loss / 100))
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d train images: %.3f %%' % (total, 100.0 * correct / total))

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
