# Gymnasium 基础用法

原文地址：<https://gymnasium.farama.org/introduction/basic_usage/>

## 什么是强化学习？

在进入 Gymnasium 之前，先明确我们要解决的问题。强化学习可以理解成一种“试错式学习”：智能体不断尝试动作，接收环境反馈的奖励，并逐步改进行为。它有点像训练宠物、通过反复练习学习骑车，或者不断玩游戏直到掌握技巧。

关键点在于：我们不会直接告诉智能体“每一步该怎么做”，而是构造一个环境，让它能够安全地探索，并从自己行为带来的后果中学习。

## 为什么使用 Gymnasium？

无论你是想训练一个会玩游戏的智能体、控制机器人，还是优化交易策略，Gymnasium 都提供了统一的工具来搭建和测试这些想法。

Gymnasium 的核心是为单智能体强化学习环境提供统一 API，并内置了许多常见环境实现，例如 `CartPole`、`Pendulum`、`MountainCar`、`MuJoCo`、`Atari` 等。本页主要介绍 Gymnasium 的基础使用方式，以及四个最关键的接口：

- `gym.make()`
- `Env.reset()`
- `Env.step()`
- `Env.render()`

Gymnasium 的核心抽象是 `Env`。它是一个高层 Python 类，用来表示强化学习中的马尔可夫决策过程（MDP，虽然并不完全等价）。借助这个类，用户可以开始新的 episode、执行动作，并可视化当前环境状态。

除了 `Env`，Gymnasium 还提供 `Wrapper`，用来在不改底层环境代码的前提下，对观测、奖励、动作等内容做增强或修改。

## 初始化环境

在 Gymnasium 中创建环境非常直接，通常使用 `make()`：

```python
import gymnasium as gym

# 创建一个适合入门的简单环境
env = gym.make("CartPole-v1")

# CartPole 的目标是在移动小车上保持杆子平衡
# - 简单，但不至于过于无聊
# - 训练速度快
# - 成功/失败条件清晰
```

这个函数会返回一个可交互的 `Env` 实例。如果你想查看当前可创建的所有环境，可以使用 `pprint_registry()`。此外，`make()` 还支持更多参数，例如传递环境关键字参数、增加或减少包装器等。

## 理解智能体-环境循环

强化学习里最经典的概念之一，就是“智能体-环境循环”：

1. 智能体观察当前状态。
2. 智能体基于观测选择一个动作。
3. 环境根据动作返回新状态和奖励。
4. 不断重复，直到本轮 episode 结束。

这个循环看起来很简单，但它正是智能体学习下棋、控制机器人、优化业务流程等能力的基础。

## 第一个 RL 程序

下面用 `CartPole` 写一个最基础的例子：

```python
# 运行前可先安装：
# pip install "gymnasium[classic-control]"

import gymnasium as gym

# 创建训练环境：一辆需要保持杆子平衡的小车
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境，开始新的 episode
observation, info = env.reset()

# observation: 智能体能看到的内容，例如小车位置、速度、杆子角度等
# info: 附加调试信息，入门阶段通常不需要重点关注
print(f"初始观测: {observation}")

# 可能类似：
# [ 0.01234567 -0.00987654 0.02345678 0.01456789]
# [小车位置, 小车速度, 杆子角度, 杆子角速度]

episode_over = False
total_reward = 0

while not episode_over:
    # 选择动作：0 表示向左推，1 表示向右推
    action = env.action_space.sample()  # 这里只是随机动作，真实训练会使用策略

    # 执行动作，观察环境反馈
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: 每多保持一帧直立，通常得到 +1
    # terminated: 因任务成功或失败而结束
    # truncated: 因时间上限等外部限制而结束
    total_reward += reward
    episode_over = terminated or truncated

print(f"本轮结束，总奖励: {total_reward}")
env.close()
```

运行时你应该会看到一个窗口，里面是一辆小车和一根杆子。由于这里用的是随机动作，小车会左右乱动，杆子最后大概率会倒下。这是正常现象，因为此时智能体还没有“学会”任何策略。

## 代码逐步解释

首先，我们用 `make()` 创建环境，并通过可选参数 `render_mode` 指定可视化方式。不同渲染模式的含义大致如下：

- `"human"`：直接打开窗口给人看。
- `"rgb_array"`：返回图像数组，适合录视频或后续处理。
- `None`：不渲染，训练通常最快。

环境初始化后，需要先调用 `Env.reset()`，才能拿到第一条观测以及额外信息。这一步相当于“开新局”。如果你想在重置时指定随机种子或选项，可以在 `reset()` 中传入 `seed` 或 `options` 参数。

因为我们不知道 episode 会在第几步结束，所以通常会定义一个变量，例如 `episode_over`，用来控制 `while` 循环。

接下来，智能体通过 `Env.step()` 执行动作。这个动作可以理解为机器人移动、手柄按键、或者一次交易决策。环境接收动作后，会返回：

- 新的观测 `observation`
- 当前动作对应的奖励 `reward`
- 是否因任务自然结束的 `terminated`
- 是否因外部限制截断的 `truncated`
- 附加信息 `info`

其中，一次动作和一次环境反馈构成一个 timestep。

环境在若干 timestep 之后可能结束，这种结束有两种常见原因：

- `terminated=True`：任务本身结束了，例如失败、成功、撞墙、到达目标等。
- `truncated=True`：达到时间上限等人为限制，并非任务逻辑自然终止。

只要其中一个为 `True`，通常就应该结束当前 episode，并在需要时再次调用 `env.reset()` 开启新一轮。

## 动作空间与观测空间

每个环境都会通过两个属性定义合法输入和输出的格式：

- `action_space`
- `observation_space`

这两个空间能告诉你：

- 智能体允许执行什么动作
- 环境会返回什么形式的观测

上面的示例中，我们调用了 `env.action_space.sample()` 来随机采样动作，而不是使用真正的策略网络。理解这些空间对编写智能体非常关键。

### 常见理解方式

- 动作空间：智能体“能做什么”，例如离散选择或连续控制值。
- 观测空间：智能体“能看到什么”，例如图像、数值向量或结构化数据。

`Env.action_space` 和 `Env.observation_space` 都是 `Space` 的实例。这个高层类常用的方法包括：

- `Space.contains()`：检查某个值是否属于该空间。
- `Space.sample()`：从该空间随机采样一个合法值。

Gymnasium 支持多种空间类型：

- `Box`：有上下界的 n 维连续空间，例如连续控制量或图像像素。
- `Discrete`：离散空间，可选值通常是 `{0, 1, ..., n-1}`。
- `MultiBinary`：n 维二值空间，例如一组开关。
- `MultiDiscrete`：多个离散空间的组合，每一维可有不同取值数量。
- `Text`：字符串空间，带最小和最大长度约束。
- `Dict`：由多个更简单空间组成的字典结构。
- `Tuple`：由多个简单空间组成的元组结构。
- `Graph`：图结构空间，包含节点与边。
- `Sequence`：长度可变的序列空间。

下面看一个简单示例：

```python
import gymnasium as gym

# 离散动作空间
env = gym.make("CartPole-v1")
print(f"动作空间: {env.action_space}")        # Discrete(2)，向左或向右
print(f"示例动作: {env.action_space.sample()}")  # 0 或 1

# 连续观测空间
print(f"观测空间: {env.observation_space}")   # 4 维 Box
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])

print(f"示例观测: {env.observation_space.sample()}")  # 一个合法的随机观测
```

## 修改环境

`Wrapper` 是修改现有环境最方便的方式之一。你可以把它理解为一层“过滤器”或“变换器”：它不需要改动底层环境实现，就能改变你与环境交互的方式。

使用包装器的好处包括：

- 避免重复写样板代码
- 保持环境结构模块化
- 可以链式组合多个包装器

通过 `gymnasium.make()` 创建的大多数环境，默认已经带有一些常见包装器，例如：

- `TimeLimit`：在超过最大步数后结束 episode
- `OrderEnforcing`：确保 `reset()` / `step()` 的调用顺序正确
- `PassiveEnvChecker`：帮助检查环境使用是否符合规范

下面是包装环境的例子：

```python
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# 创建一个观测比较复杂的环境
env = gym.make("CarRacing-v3")
print(env.observation_space.shape)  # (96, 96, 3)，96x96 RGB 图像

# 用包装器把观测压平成一维数组
wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space.shape)  # (27648,)

# 这样做对于某些只接受一维输入的算法更方便
```

初学者常会用到的包装器包括：

- `TimeLimit`：当 timestep 超过上限时发出 `truncated` 信号，防止 episode 无限持续。
- `ClipAction`：将传给 `step()` 的动作裁剪到合法范围内。
- `RescaleAction`：把动作重新缩放到另一个区间，例如算法输出在 `[-1, 1]`，但环境要求 `[0, 10]`。
- `TimeAwareObservation`：把当前时间步信息加入观测，有时能帮助学习。

如果你已经拿到一个被多层包装的环境，但想访问其最原始的底层环境，可以使用 `unwrapped` 属性：

```python
print(wrapped_env)
# <FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>>

print(wrapped_env.unwrapped)
# <gymnasium.envs.box2d.car_racing.CarRacing object at ...>
```

## 初学者常见问题

### 智能体行为方面

- 如果智能体看起来完全随机，这很正常，因为 `env.action_space.sample()` 本来就是随机动作。真正的学习发生在你用策略来替代随机采样之后。
- 如果 episode 一开始就结束，要检查是否正确处理了每一轮之间的 `reset()`。

### 常见代码错误

错误写法：

```python
env = gym.make("CartPole-v1")
obs, reward, terminated, truncated, info = env.step(action)  # 会报错
```

正确写法：

```python
env = gym.make("CartPole-v1")
obs, info = env.reset()  # 必须先 reset
obs, reward, terminated, truncated, info = env.step(action)
```

## 下一步

掌握这些基础后，可以继续做下面几件事：

- 训练一个真正的智能体：把随机动作替换成策略。
- 自定义环境：构建自己的强化学习任务。
- 记录智能体行为：保存训练过程中的视频和数据。
- 加速训练：使用向量化环境等优化手段。
