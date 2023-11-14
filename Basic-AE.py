import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)  # dropout
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(x)
        x = self.add_noise(x)  # 添加噪声
        x = self.dropout(x)     # dropout
        x = self.decoder(x)
        return x

    def add_noise(self, inputs, noise_factor=0.5):
        noise = torch.randn_like(inputs) * noise_factor # 定义噪声的强度
        return inputs + noise


# 定义训练函数
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.view(inputs.size(0), -1)  # 展平输入图像
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 定义数据集和数据加载器
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

# 创建自动编码器模型
input_size = 28 * 28  # MNIST图像大小
hidden_size = 256
autoencoder = Autoencoder(input_size, hidden_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练自动编码器
train_autoencoder(autoencoder, dataloader, criterion, optimizer, num_epochs=10)
