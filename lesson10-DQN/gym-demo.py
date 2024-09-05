import gym

# 创建环境
env = gym.make('CartPole-v1')

env.reset()

for _ in range(1000):
    env.render()  # 显示当前环境
    action = env.action_space.sample()  # 随机选择一个动作
    state, reward, done, truncated, info = env.step(action)  # 执行动作
    if done or truncated:
        env.reset()  # 如果结束，重置环境

env.close()

# gym是一个强化学习的环境库，提供了很多强化学习的环境，可以用来测试强化学习算法。