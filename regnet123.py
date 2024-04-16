import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
import time
# 定义ConvRNN模块
class ConvRNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvRNN, self).__init__()
        self.conv = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=1)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = self.conv(combined)
        return h

# 修改Block类以集成ConvRNN模块
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # 添加ConvRNN模块
        self.convrnn = ConvRNN(out_channels, out_channels, kernel_size=3)

    def forward(self, x, h=None):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)


        if h is not None:
            out = self.convrnn(out, h)

        return out

class RegNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(RegNet, self).__init__()
        # Simplified RegNet settings
        layers = [2, 4, 6, 2]  # Example configuration, should be adjusted based on Y parameter
        channels = [64, 128, 256, 512]  # Simplified, should be derived from the RegNet formula

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[0], channels[1], layers[0], stride=2)
        self.layer2 = self._make_layer(channels[1], channels[2], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[3], channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [Block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
import torchvision
import torchvision.transforms as transforms

def load_cifar100(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader




def train_and_evaluate(model, trainloader, testloader, epochs=150):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    total_start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录每个 epoch 开始时间
        model.train()
        train_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / len(trainloader.dataset)
        epoch_val_loss = val_loss / len(testloader.dataset)
        acc = 100 * correct / total
        print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Accuracy: {acc}%')

        # 如果当前准确率高于之前所有epoch的准确率，则保存模型
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

        epoch_end_time = time.time()  # 记录每个 epoch 结束时间
        epoch_time_elapsed = epoch_end_time - epoch_start_time  # 计算每个 epoch 所花费的时间
        remaining_time = epoch_time_elapsed * (epochs - epoch - 1)  # 计算剩余的预计完成时间

        print(f'Epoch {epoch + 1} took {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s, Estimated remaining time: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s')

    total_end_time = time.time()  # 记录训练结束时间
    total_time_elapsed = total_end_time - total_start_time  # 计算总体花费的时间
    print(f'Training complete in {total_time_elapsed // 60:.0f}m {total_time_elapsed % 60:.0f}s')
    print(f'Best Accuracy: {best_acc}%')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    # 保存最佳模型
    torch.save(model.state_dict(), 'best_model.pth')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegNet(num_classes=100).to(device)  # 注意修改 num_classes
    trainloader, testloader = load_cifar100(batch_size=64)  # 加载 CIFAR-100 数据集
    train_and_evaluate(model, trainloader, testloader)
