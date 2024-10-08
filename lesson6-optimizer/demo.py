import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# torch.manual_seed(1)  # reproducible

LR = 0.01
BATCH_SIZE = 32
EPOCH = 10

# fake dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

class Net(torch.nn.Module):
    def __init__(self, n_feature=1, n_hidden=10, n_output=1):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(in_features=n_feature, out_features=n_hidden)
        self.output = torch.nn.Linear(in_features=n_hidden, out_features=n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
    
if __name__=='__main__':
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_Adam.parameters(),lr=LR,alpha=0.9)
    opt_Adam = torch.optim.Adam(net_RMSprop.parameters(),lr=LR,betas=(0.9,0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses = [[], [], [], []]   # record loss

    # training
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        for step, (b_x, b_y) in enumerate(loader):          # for each training step
            for net, opt, loss_his in zip(nets, optimizers, losses):
                output = net(b_x)              # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()                # clear gradients for next train
                loss.backward()                # backpropagation, compute gradients
                opt.step()                     # apply gradients
                loss_his.append(loss.data.numpy())     # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, loss_his in enumerate(losses):
        plt.plot(loss_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()