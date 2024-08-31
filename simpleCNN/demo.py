# 四层CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义网络
class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN,self).__init__()
    # 卷积层
    self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
    self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
    self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
    self.conv4=nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
    # 最大池化层
    self.pool=nn.MaxPool2d(2,2)
    # FC全连接
    self.fc1=nn.Linear(256,512)
    self.fc2=nn.Linear(512,10)

  def forward(self, x):
      # 卷积激活池化
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = self.pool(F.relu(self.conv4(x)))
      
      # 动态展平
      x = x.view(x.size(0), -1)
      # x = x.view(-1, 256*3*3)
      # 全连接层
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x


# 加载数据集  
transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,),(0.3081,))
])
train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=1000,shuffle=False)

# 定义训练过程
def train(model,device,train_loader,optimizer,epoch):
  model.train()
  for batch_index,(data,target) in enumerate(train_loader):
    data,target=data.to(device),target.to(device)
    optimizer.zero_grad()
    output=model(data)
    #
    loss=F.cross_entropy(output,target)
    loss.backward()
    optimizer.step()
    if batch_index %100==0:
      print(f'Train Epoch: {epoch} [{batch_index*len(data)}/{len(train_loader)}]',f'Loss: {loss.item():.6f}')

def test(model,device,test_loader):
  model.eval()
  test_loss=0
  correct=0
  with torch.no_grad():
    for data,target in test_loader:
      data,target=data.to(device),target.to(device)
      output=model(data)
      test_loss+=F.cross_entropy(output,target,reduction='sum').item()
      pred=output.argmax(dim=1,keepdim=True)
      correct+=pred.eq(target.view_as(pred)).sum().item()

  test_loss/=len(test_loader.dataset)
  accuracy=100.*correct/len(test_loader.dataset)
  print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} 'f'({accuracy:.2f}%)\n')

device=torch.device('cpu')
model=SimpleCNN().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.01)

for epoch in range(1,11):
  train(model,device,train_loader,optimizer,epoch)
  test(model,device,test_loader)