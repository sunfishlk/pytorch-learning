import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
from matplotlib import cm
from sklearn.manifold import TSNE

HAS_SK = True
torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 2
BATCH_SIZE = 50
LR = 0.001          # learning rate
DOWNLOAD_MNIST = False

# 训练集  算loss
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),   # 把图像转换成张量
    download=DOWNLOAD_MNIST,
)

# plot one example
# print(train_data.data.size())                 # (60000, 28, 28)
# print(train_data.targets.size())               # (60000)
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

# 批处理训练 the image batch shape will be (50, 1, 28, 28) , 50 images, 1 channel, 28x28 (50, 1, 28, 28) 一批处理50张图片，每张是(1,28,28)黑白照片
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 测试集  算准确率要用

test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor())
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255  # test_data has been renamed data 测试集太大，60000张，所以只取前2000张
test_y = test_data.targets[:2000]  # test_labels has been renamed targets

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 输入是(1,28,28) 对应(通道/高度, 长, 宽)
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),                             # 卷积后是(16,28,28)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # 池化后(16,14,14)，用2x2向下采样
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(               # 输入(16,14,14)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2  # padding=(kernel_size-1)/2, stride=1, 不改变图片张量的长宽
            ),                             # 输出(32,14,14)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)  # 池化后(32,7,7)
        )
        # 全连接
        self.out = torch.nn.Linear(32*7*7, 10)  # 全连接线性连接，一个输入一个输出，这里MNIST数据集分类，10种，所以输出是10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 两次卷积后为(32,7,7),展平为(batch_size,7,7)
        x = x.view(x.size(0), -1)
        prediction = self.out(x)
        return prediction  # prediction为0~9, 10种

# 实例化网络模型
cnn = CNN()
# print(cnn)

# 优化器和损失
loss_func = torch.nn.CrossEntropyLoss()  # 分类，交叉熵损失
# optimizer = torch.optim.SGD(params=cnn.parameters(), lr=LR)  # SGD simplest
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=LR)  # Adam highest

# 引入画图功能
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max())
#     plt.ylim(Y.min(), Y.max())
#     plt.title('Visualize last layer')
#     plt.show()
#     plt.pause(0.01)

# plt.ion()
# plt.show()

# 训练
def train():
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # 计算训练集的loss
            out = cnn(batch_x)
            loss = loss_func(out, batch_y)
            # 清除梯度，反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每50步(批处理大小batch_size)打印一次训练集的loss和测试集的准确率
            if step % 50 == 0:
                # 计算测试集的准确率
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = (pred_y == test_y).sum().item() / test_y.size(0)
                print('Epoch:', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
                # if HAS_SK:
                #     # Visualization of trained flatten layer (T-SNE)
                #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                #     plot_only = 500
                #     low_dim_embs = tsne.fit_transform(out.data.numpy()[:plot_only, :])
                #     labels = batch_y.numpy()[:plot_only]
                #     plot_with_labels(low_dim_embs, labels)
    # plt.ioff()
            
if __name__ == '__main__':
    train()
    # plt.ioff()
    # plt.show()