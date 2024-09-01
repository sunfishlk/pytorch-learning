import torch
import torch.utils.data as Data

def main():
    torch.manual_seed(1)  # reproducible，设置随机数种子，每次生成随机数都是相同种子，保证每次运行结果一致
    # BATCH_SIZE = 8  # 批训练的数据个数，8个为一批，算作一个小样本
    BATCH_SIZE = 3

    # make fake data
    x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
    y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # 训练3轮
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print('Epoch: ', epoch, ' | Step: ', step, ' | batch x ', batch_x.data.numpy(), ' | batch_y ', batch_y.data.numpy())

if __name__ == '__main__':
    main()