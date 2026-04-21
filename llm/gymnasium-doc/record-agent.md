# 录制智能体行为

原文地址：<https://gymnasium.farama.org/introduction/record_agent/>

## 为什么要录制智能体？

在强化学习开发中，录制智能体行为非常有价值，原因通常有这些：

- 可视化理解：直接看到智能体到底在做什么。很多时候，一个 10 秒视频比盯着奖励曲线看半天更容易发现问题。
- 性能跟踪：系统性地收集 episode 奖励、长度、耗时等数据，观察训练进展。
- 调试：定位具体失败模式、异常行为，或者找出智能体在哪些环境里表现差。
- 评估：对不同算法、训练轮次、超参数组合进行客观比较。
- 沟通展示：方便和协作者分享结果，也适合做论文图示或教学演示。

## 什么时候录制？

### 在评估阶段

评估一个已经训练好的智能体时，通常适合录制每一个 episode：

- 查看最终表现
- 生成演示视频
- 分析具体行为细节

### 在训练阶段

训练时 episode 数量往往非常多，不适合全部录下来，通常按周期录制：

- 观察学习过程是否在改进
- 尽早发现训练异常
- 做出“学习过程延时视频”

Gymnasium 主要提供两个相关包装器：

- `RecordEpisodeStatistics`：记录数值统计信息，例如总奖励、episode 长度、耗时
- `RecordVideo`：基于环境渲染结果生成 MP4 视频

这篇文档主要介绍两种典型用法：

- 在评估阶段录制每个 episode
- 在训练阶段周期性录制

## 录制每个 Episode（评估场景）

在评估一个训练好的智能体时，通常希望连续运行若干个 episode，观察平均表现和稳定性。这时可以把 `RecordVideo` 和 `RecordEpisodeStatistics` 一起用上。

```python
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np

# 配置
num_eval_episodes = 4
env_name = "CartPole-v1"  # 替换成你的环境

# 创建支持录制的环境
env = gym.make(env_name, render_mode="rgb_array")

# 每个 episode 都录视频
env = RecordVideo(
    env,
    video_folder="cartpole-agent",
    name_prefix="eval",
    episode_trigger=lambda x: True,
)

# 记录 episode 统计信息
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

print(f"Starting evaluation for {num_eval_episodes} episodes...")
print("Videos will be saved to: cartpole-agent/")

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    episode_over = False

    while not episode_over:
        # 这里应替换为你自己的智能体策略
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        episode_over = terminated or truncated

    print(
        f"Episode {episode_num + 1}: "
        f"{step_count} steps, reward = {episode_reward}"
    )

env.close()

# 打印统计摘要
print("\nEvaluation Summary:")
print(f"Episode durations: {list(env.time_queue)}")
print(f"Episode rewards: {list(env.return_queue)}")
print(f"Episode lengths: {list(env.length_queue)}")

# 计算一些常用指标
avg_reward = np.sum(env.return_queue)
avg_length = np.sum(env.length_queue)
std_reward = np.std(env.return_queue)

print(f"\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"Average episode length: {avg_length:.1f} steps")
print(
    f"Success rate: "
    f"{sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}"
)
```

### 关键点说明

- `render_mode="rgb_array"` 是录视频所必需的
- `episode_trigger=lambda x: True` 表示每个 episode 都录制
- `RecordEpisodeStatistics` 会把统计信息放进内部队列，后面可以直接读取

## 如何理解输出结果

运行上面的代码后，通常会得到两类结果。

### 视频文件

例如：

- `cartpole-agent/eval-episode-0.mp4`
- `cartpole-agent/eval-episode-1.mp4`

每个文件对应一个完整 episode，适合：

- 看清楚智能体到底怎么行动
- 做演示
- 逐帧排查异常行为

### 控制台输出

例如：

```text
Episode 1: 23 steps, reward = 23.0
Episode 2: 15 steps, reward = 15.0
Episode 3: 200 steps, reward = 200.0
Episode 4: 67 steps, reward = 67.0

Average reward: 76.25 ± 78.29
Average episode length: 76.2 steps
Success rate: 100.0%
```

### 统计队列

`RecordEpisodeStatistics` 会维护几个很实用的队列：

- `env.time_queue`：每个 episode 的实际耗时
- `env.return_queue`：每个 episode 的总奖励
- `env.length_queue`：每个 episode 的步数

这些数据可以直接拿来做平均值、标准差、成功率等统计分析。

文档也提到，如果评估阶段非常耗时，可以使用向量化环境并行评估多个 episode，而不是串行一个个跑。

## 在训练过程中录制（周期性录制）

训练阶段往往会运行成百上千个 episode，所以一般不会每一局都录制，而是每隔一段时间录一次。

```python
import logging
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# 训练配置
training_period = 250          # 每 250 个 episode 录一次视频
num_training_episodes = 10_000
env_name = "CartPole-v1"

# 配置日志输出
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 创建环境
env = gym.make(env_name, render_mode="rgb_array")

# 周期性录视频
env = RecordVideo(
    env,
    video_folder="cartpole-training",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0,
)

# 每个 episode 都记录统计信息
env = RecordEpisodeStatistics(env)

print(f"Starting training for {num_training_episodes} episodes")
print(f"Videos will be recorded every {training_period} episodes")
print("Videos saved to: cartpole-training/")

for episode_num in range(num_training_episodes):
    obs, info = env.reset()
    episode_over = False

    while not episode_over:
        # 替换成你的真实训练智能体
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    # episode 结束后，info 中会带上统计信息
    if "episode" in info:
        episode_data = info["episode"]
        logging.info(
            f"Episode {episode_num}: "
            f"reward={episode_data['r']:.1f}, "
            f"length={episode_data['l']}, "
            f"time={episode_data['t']:.2f}s"
        )

    # 定期做额外分析
    if episode_num % 1000 == 0:
        recent_rewards = list(env.return_queue)[-100:]
        if recent_rewards:
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            print(
                f" -> Average reward over last 100 episodes: {avg_recent:.1f}"
            )

env.close()
```

这种方式的重点是：

- 视频按周期保存，不会占满磁盘
- 每个 episode 的统计信息依然持续记录
- 可以结合最近若干局的平均奖励观察训练是否在进步

## 训练过程录制的价值

周期性录制最直接的用途，是观察智能体随着训练逐步变好的过程。例如：

- `training-episode-0.mp4`：几乎完全随机
- `training-episode-250.mp4`：开始出现一点规律
- `training-episode-500.mp4`：明显有改进
- `training-episode-1000.mp4`：已经具备较稳定表现

除了视频，还可以把统计数据画成学习曲线。

```python
import matplotlib.pyplot as plt

episodes = range(len(env.return_queue))
rewards = list(env.return_queue)

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, alpha=0.3, label="Episode Rewards")

# 增加滑动平均，便于看趋势
window = 100
if len(rewards) > window:
    moving_avg = [
        sum(rewards[i:i + window]) / window
        for i in range(len(rewards) - window + 1)
    ]
    plt.plot(
        range(window - 1, len(rewards)),
        moving_avg,
        label=f"{window}-Episode Moving Average",
        linewidth=2,
    )

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Learning Progress")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

滑动平均可以把训练中的噪声平滑掉，让总体趋势更容易看清。

## 与实验跟踪工具集成

如果项目更正式，通常会把统计结果和视频进一步接到实验跟踪工具里，比如 `Weights & Biases`。

```python
import os
import wandb

wandb.init(project="cartpole-training", name="q-learning-run-1")

for episode_num in range(num_training_episodes):
    # ... training code ...

    if "episode" in info:
        episode_data = info["episode"]
        wandb.log(
            {
                "episode": episode_num,
                "reward": episode_data["r"],
                "length": episode_data["l"],
                "episode_time": episode_data["t"],
            }
        )

    if episode_num % training_period == 0:
        video_path = f"cartpole-training/training-episode-{episode_num}.mp4"
        if os.path.exists(video_path):
            wandb.log({"training_video": wandb.Video(video_path)})
```

这样可以把：

- 奖励曲线
- episode 长度
- 训练耗时
- 周期性视频

都统一放到实验平台里做长期追踪和对比。

## 最佳实践总结

### 用于评估时

- 尽量录制每个 episode，获得完整表现画像
- 使用多个随机种子，增强统计意义
- 同时保存视频和数值数据
- 对关键指标计算均值和波动范围

### 用于训练时

- 按周期录制，例如每 `100-1000` 个 episode 录一次
- 训练期间更应关注数值统计，视频主要用于抽查
- 可以使用更灵活的触发条件，只录制关键 episode
- 长时间训练时注意磁盘和内存占用

### 用于分析时

- 使用滑动平均平滑高噪声学习曲线
- 同时分析成功案例和失败案例
- 比较智能体在训练不同阶段的行为变化
- 保留原始数据，便于后续复盘和横向比较

## 更多信息

如果要继续往下学，这几个方向最自然：

- 训练智能体：先真正训练出一个值得录制的 agent
- 基础用法：补足 Gymnasium 的核心 API 使用方式
- 更多训练教程：了解更完整的算法与流程
- 自定义环境：构造你自己的可录制环境

录制智能体行为不是附属功能，而是强化学习实践里非常关键的一环。它能帮助你理解智能体到底学到了什么、及时发现训练问题，并把结果用更直观的方式表达出来。
