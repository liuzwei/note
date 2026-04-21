# 训练智能体

原文地址：<https://gymnasium.farama.org/introduction/train_agent/>

## 强化学习中的“训练”是什么意思？

当我们说训练一个强化学习智能体时，本质上是在教它通过经验做出更好的决策。和监督学习不同，监督学习会直接提供“标准答案”，而强化学习中的智能体需要自己尝试不同动作，再根据结果好坏逐步调整行为。

这有点像学骑自行车：你会不断尝试不同动作，摔几次，修正几次，最后慢慢知道什么做法更有效。

训练的目标是学出一个策略（policy）：它告诉智能体在每一种状态下应该采取什么动作，才能让长期累计奖励最大化。

## 直观理解 Q-Learning

这篇教程用 Q-learning 来求解 `Blackjack-v1` 环境。先不急着看公式，可以先从概念上理解它。

Q-learning 会构造一张巨大的“经验备忘表”，也就是 Q-table，用来记录：

- 每种状态下可以执行哪些动作
- 每个动作在该状态下有多“值钱”

你可以把它理解为：

- 行：智能体可能遇到的不同状态
- 列：智能体可以执行的不同动作
- 值：在该状态下执行该动作的预期未来回报

对于 Blackjack：

- 状态：玩家当前点数、庄家明牌、是否有可用 Ace
- 动作：`Hit`（继续要牌）或 `Stand`（停牌）
- Q 值：在某个状态下执行某个动作后，预期能拿到的回报

### 学习过程

Q-learning 的学习过程可以概括成几步：

1. 先执行一个动作，看看结果如何。
2. 再修正“经验表”中的估计，判断这个动作比原先想象中更好还是更差。
3. 反复试错，不断更新估计。
4. 在“探索新动作”和“利用已知最优动作”之间做平衡。

它之所以有效，是因为随着训练进行，好的动作会在表中得到更高的 Q 值，坏的动作会得到更低的 Q 值。最终，智能体会倾向于选择预期回报最高的动作。

这篇页面给出了一个在 Gymnasium 中训练智能体的简要示例：使用表格型 Q-learning 求解 `Blackjack-v1`。如果你想看更完整的环境与算法教程，可以再去看官方的训练类教程页面。阅读这篇前，建议先看基础用法。

## 关于环境：Blackjack

Blackjack 是赌场里非常经典的纸牌游戏，也很适合作为强化学习入门环境，因为它具备这些特点：

- 规则清晰：在不爆牌的前提下，让点数尽可能接近 21，并且超过庄家。
- 观测简单：玩家点数、庄家明牌、是否有可用 Ace。
- 动作离散：只需要决定要牌还是停牌。
- 反馈直接：每一局结束后立刻知道输赢或平局。

这个版本使用的是“无限牌堆”（有放回抽样），所以数牌策略无效，智能体只能通过反复试错学习最优基础策略。

### 环境细节

- 观测：`(player_sum, dealer_card, usable_ace)`
- `player_sum`：玩家当前手牌点数，范围通常为 `4-21`
- `dealer_card`：庄家明牌，范围为 `1-10`
- `usable_ace`：玩家是否持有可作为 11 使用的 Ace，布尔值
- 动作：`0 = Stand`，`1 = Hit`
- 奖励：赢 `+1`，输 `-1`，平局 `0`
- episode 结束条件：玩家停牌，或者爆牌（点数超过 21）

## 执行动作

在我们通过 `env.reset()` 拿到第一条观测之后，就可以使用 `env.step(action)` 和环境交互。

```python
observation, reward, terminated, truncated, info = env.step(action)
```

这个函数会返回五个关键值：

- `observation`：执行动作后的新状态，也就是智能体下一步能看到的内容
- `reward`：该动作带来的即时反馈，在 Blackjack 中通常是 `+1`、`-1` 或 `0`
- `terminated`：该 episode 是否自然结束，例如这一手牌已经结束
- `truncated`：是否因为时间上限等外部条件被截断，Blackjack 中通常不会用到
- `info`：额外调试信息，很多基础场景下可以先忽略

关键点在于：`reward` 只告诉你当前动作的即时好坏，但智能体真正要学的是“长期后果”。Q-learning 通过估计未来累计回报来解决这个问题，而不是只看当前这一步的奖励。

## 构建一个 Q-Learning 智能体

要实现一个可训练的智能体，至少需要几部分能力：

- 选择动作
- 根据经验更新 Q 值
- 管理探索率，让随机性随训练逐渐下降

## 探索与利用

这是强化学习里最基础也最重要的矛盾之一：

- 探索（exploration）：尝试新的动作，了解环境
- 利用（exploitation）：优先使用当前已知效果最好的动作

这里使用的是 epsilon-greedy 策略：

- 以 `epsilon` 的概率随机选动作，用于探索
- 以 `1 - epsilon` 的概率选择当前 Q 值最大的动作，用于利用

一般做法是：

- 训练初期让 `epsilon` 较高，鼓励多探索
- 随训练推进逐步减小 `epsilon`，让智能体更多利用已学到的知识

下面是官方示例中的智能体实现：

```python
from collections import defaultdict

import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """初始化一个 Q-Learning 智能体。

        Args:
            env: 训练环境
            learning_rate: Q 值更新速度，范围通常为 0-1
            initial_epsilon: 初始探索率，通常设为 1.0
            epsilon_decay: 每个 episode 衰减多少探索率
            final_epsilon: 最小探索率，通常设为 0.1
            discount_factor: 未来奖励折扣因子
        """
        self.env = env

        # Q 表：把状态映射到每个动作对应的预期回报
        # defaultdict 会在遇到新状态时自动创建全 0 向量
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # 探索相关参数
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # 记录训练误差，便于后续分析
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """使用 epsilon-greedy 策略选择动作。"""
        # 以 epsilon 概率探索
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # 否则利用当前最优动作
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """根据经验更新 Q 值。

        这里使用的是典型的 Q-learning 更新逻辑：
        (state, action, reward, next_state)
        """
        # 如果 episode 已结束，则未来回报为 0
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # 贝尔曼方程目标值
        target = reward + self.discount_factor * future_q_value

        # 当前估计误差
        temporal_difference = target - self.q_values[obs][action]

        # 按学习率向目标值靠近
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # 保存训练误差，便于可视化
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """每个 episode 后衰减探索率。"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

## 理解 Q-Learning 的更新公式

学习的核心发生在 `update()` 方法里。可以把它拆成下面几步：

```python
# 当前估计值：Q(state, action)
current_q = self.q_values[obs][action]

# 实际观测到的目标值：即时奖励 + 折扣后的未来最优价值
target = reward + self.discount_factor * max(self.q_values[next_obs])

# 当前估计错了多少
error = target - current_q

# 向目标值靠近一步
new_q = current_q + learning_rate * error
```

这就是经典贝尔曼方程在起作用。它的核心思想是：

一个状态-动作对的价值，应该等于“当前得到的即时奖励”加上“未来最优动作价值的折扣和”。

## 训练智能体

接下来进入真正的训练过程。整体流程如下：

1. 调用 `reset()` 开启一个新的 episode。
2. 在这个 episode 中不断选择动作、与环境交互并更新 Q 值。
3. episode 结束后，降低探索率。
4. 重复很多轮，直到策略稳定。

先定义超参数和环境：

```python
# 训练超参数
learning_rate = 0.01      # 学习速度，越大更新越快，但也可能更不稳定
n_episodes = 100_000      # 训练的总局数
start_epsilon = 1.0       # 初始为 100% 随机探索
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1       # 保留最小探索率

# 创建环境与智能体
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
```

### 训练循环

```python
from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    # 开始新的一局
    obs, info = env.reset()
    done = False

    # 完整打完一局
    while not done:
        # 选择动作：一开始偏随机，后面逐渐更依赖已学策略
        action = agent.get_action(obs)

        # 执行动作并观察结果
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 用这次经验更新 Q 值
        agent.update(obs, action, reward, terminated, next_obs)

        # 切换到下一个状态
        done = terminated or truncated
        obs = next_obs

    # 逐步降低探索率
    agent.decay_epsilon()
```

## 训练过程中会看到什么？

在不同训练阶段，智能体的表现通常会有明显差异：

### 早期阶段（0 到 10,000 局）

- 智能体基本是随机行动，因为 `epsilon` 很高
- 胜率大约在 43% 左右
- 训练误差较大，说明 Q 值估计还很不准确

### 中期阶段（10,000 到 50,000 局）

- 智能体开始学到一些有效策略
- 胜率逐渐提升到 45% 到 48%
- 训练误差开始下降

### 后期阶段（50,000 局以后）

- 智能体逐渐收敛到接近最优策略
- 胜率通常稳定在约 49% 左右
- 训练误差较小，Q 值基本趋于稳定

## 分析训练结果

训练完成后，通常会把一些统计量画出来观察趋势。官方示例使用下面的代码：

```python
from matplotlib import pyplot as plt


def get_moving_avgs(arr, window, convolution_mode):
    """计算滑动平均，用于平滑噪声。"""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


# 使用 500 个 episode 的窗口做平滑
rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# 每局奖励
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# 每局长度
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# 训练误差
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()
```

## 如何解读结果图

### 奖励曲线

奖励曲线一般会从大约 `-0.05` 逐步改善到 `-0.01` 左右。Blackjack 并不是一个“能稳定正收益”的简单游戏，即使策略接近最优，也通常会受到庄家优势影响。

### episode 长度

每局动作数一般会稳定在 `2-3` 步左右：

- 如果过短，可能说明智能体过早停牌
- 如果过长，可能说明智能体过于频繁要牌

### 训练误差

训练误差应当随着时间逐渐减小，这表示智能体的价值估计越来越准确。训练初期出现较大的尖峰是正常现象，因为智能体还在不断接触新状态。

## 常见训练问题与解决思路

### 智能体完全没有进步

现象：

- 奖励基本不变
- 训练误差很大

可能原因：

- 学习率过高或过低
- 奖励设计不合理
- Q 值更新逻辑存在 bug

可尝试的解决方式：

- 把学习率调到 `0.001` 到 `0.1` 之间测试
- 检查奖励是否符合预期，例如 Blackjack 常见设置是 `-1 / 0 / +1`
- 确认 Q-table 的值确实在更新

### 训练不稳定

现象：

- 奖励剧烈波动
- 长时间无法收敛

可能原因：

- 学习率过高
- 探索不足

可尝试的解决方式：

- 降低学习率，例如从 `0.1` 降到 `0.01`
- 确保最小探索率足够，例如 `final_epsilon >= 0.05`
- 增加训练 episode 数量

### 智能体卡在较差策略

现象：

- 很早就停止提升
- 最终表现明显低于预期

可能原因：

- 探索时间不够
- 学习率太低

可尝试的解决方式：

- 放慢 `epsilon` 衰减速度，让探索持续更久
- 在初期提高学习率
- 改用其他探索策略，例如乐观初始化

### 学习太慢

现象：

- 智能体在进步，但非常缓慢

可能原因：

- 学习率过低
- 探索过多

可尝试的解决方式：

- 适当提高学习率，同时观察是否带来不稳定
- 加快 `epsilon` 衰减速度
- 针对困难状态做更聚焦的训练

## 测试训练好的智能体

训练结束后，应该关闭探索，用纯利用模式测试智能体的真实表现。官方示例代码如下：

```python
def test_agent(agent, env, num_episodes=1000):
    """在不学习、不探索的条件下测试智能体表现。"""
    total_rewards = []

    # 临时关闭探索
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # 恢复原探索率
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


# 测试智能体
test_agent(agent, env)
```

在 Blackjack 中，比较合理的测试结果通常是：

- 胜率：约 `42% - 45%`
- 平均奖励：约 `-0.02` 到 `+0.01`
- 标准差较低：说明策略相对稳定

需要注意的是，由于庄家优势存在，胜率超过 50% 通常并不现实。

## 下一步

如果你已经完成这篇教程，可以继续尝试：

- 体验其他环境，例如 `CartPole`、`MountainCar`、`LunarLander`
- 调整超参数，例如学习率、探索率和衰减策略
- 实现其他算法，例如 `SARSA`、`Expected SARSA`、Monte Carlo 方法
- 引入函数逼近，例如使用神经网络处理更大的状态空间
- 自定义环境，设计自己的强化学习问题

这篇教程的核心启发是：强化学习智能体通过不断试错来积累“什么动作在什么状态下更有效”的知识，而 Q-learning 提供了一套系统化的方法，让这种知识可以被逐步估计出来。

如果后面你继续补这一目录，下一篇自然衔接的就是“创建自定义环境”。
