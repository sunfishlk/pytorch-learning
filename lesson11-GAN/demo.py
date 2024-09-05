import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

# 超参数
BATCH_SIZE = 64
LR_G = 0.0001  # 生成器Generator学习率  对抗学习
LR_D = 0.0001  # 判别器Discriminator学习率  对抗学习
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a*np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

G = torch.nn.Sequential(
    torch.nn.Linear(N_IDEAS, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, ART_COMPONENTS)
)

D = torch.nn.Sequential(
    torch.nn.Linear(ART_COMPONENTS, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid()
)

# 优化器
G_opt = torch.optim.Adam(D.parameters(),lr=LR_G)
D_opt = torch.optim.Adam(D.parameters(),lr=LR_D)

# 打开交互模式
plt.ion()

# 生成对抗训练 同时训练生成器和鉴别器
for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = torch.rand(BATCH_SIZE, N_IDEAS)  #生成随机数张量 作为生成器的灵感
    G_paintings = G(G_ideas)

    prob_artist1 = D(G_paintings)  # 喂鉴别器生成的假画
    G_loss = torch.mean(torch.log(1. - prob_artist1))  
    G_opt.zero_grad()
    G_loss.backward()
    G_opt.step()

    prob_artist0 = D(artist_paintings)  # 喂给鉴别器名画
    prob_artist1 = D(G_paintings.detach())  #喂给鉴别器生成器生成的假画
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))

    D_opt.zero_grad()
    D_loss.backward(retain_graph=True)
    D_opt.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()