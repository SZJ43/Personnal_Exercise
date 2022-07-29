## 2022.7.23

### nn.Conv2d()函数的运用
![c73de25feddb325bcb3e723a7022214](https://user-images.githubusercontent.com/64791841/180587070-8f09b802-61be-42d8-a08b-37e25d1a33bf.jpg)


## 2022.7.29
### 网络设计思路的总结
  在设计网络的过程中，从一开始的30%左右的准确率，通过逐渐地修改神经网络的结构和超参数等步骤；将模型在测试集的准确率最终提升到了85%左右，如下图所示：
  
  ![9d532bab0599338ddbb6b914e5a8adc](https://user-images.githubusercontent.com/64791841/181664062-ef44b152-250a-454c-b541-79834f4404f8.jpg)

以下是具体模块的代码

#### 1、对cifar-10数据集图像的预处理
      transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#### 2、网络结构

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
        
 #### 3、组装过程
 
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
      
 #### 4、超参数的设置和输出正确率
 
      optimizer = optim.Adam(self.parameters(), lr=0.0005) # 学习率设为0.0005
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
          l = loss(outputs, labels)
          l.backward()
          optimizer.step()


          # print statistics
          running_loss += l.item()
          if i == 99:  # i代表iteration的序列数
              print('[%d, %5d] loss: %.4f' %
                    (epoch, i, running_loss / 100))
              running_loss = 0.0
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              print('Accuracy of the network on the %d train images: %.3f %%' % (total, 100.0 * correct / total))
     
     
