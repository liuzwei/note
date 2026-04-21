# 创建自定义环境

原文地址：<https://gymnasium.farama.org/introduction/create_custom_env/>

## 写代码前：先做环境设计

创建一个强化学习环境，本质上和设计一个小游戏或模拟器很像。真正动手写代码前，先把学习问题本身想清楚，这一步非常关键。环境设计如果有问题，再好的算法也可能学不出来。

## 关键设计问题

在开始之前，至少要回答下面这些问题：

- 智能体要学什么能力？
- 它需要看到哪些信息？
- 它能执行哪些动作？
- 成功应该如何衡量？
- 一个 episode 应该在什么时候结束？

可以进一步具体化成这些方向：

- 技能目标：走迷宫、平衡系统、资源分配优化、博弈决策等
- 观测信息：位置、速度、系统状态、历史信息、部分可观测还是完全可观测
- 动作形式：离散动作、连续控制、多动作联合控制
- 成功标准：到达目标、最短时间、最小能耗、最高得分、避免失败
- 结束条件：任务完成、失败、超时、安全约束触发

## GridWorld 示例设计

这篇教程使用一个简单的 `GridWorld` 作为例子：

- 目标技能：高效移动到目标位置
- 可观测信息：智能体位置和目标位置
- 动作：上、下、左、右移动
- 成功标准：尽量少步数到达目标
- episode 结束：到达目标，或者可选地设置时间上限

这个问题足够简单，便于理解，又不至于完全没有挑战。

本页展示了如何在 Gymnasium 中完整实现一个自定义环境。官方也建议先熟悉基础用法，再阅读这篇内容。

这个 `GridWorld` 是一个固定大小的二维方格。每个时间步里，智能体都可以在网格中上下或左右移动一格。目标位置会在每个 episode 开始时随机放置。

## 环境 `__init__`

和所有 Gymnasium 环境一样，自定义环境需要继承 `gymnasium.Env`。其中一个硬性要求是：你需要定义动作空间和观测空间，用来声明该环境允许哪些动作输入，以及会返回什么样的观测。

根据上面的设计：

- 动作有四种，因此使用 `Discrete(4)`
- 观测用字典表示，形如 `{"agent": array([1, 0]), "target": array([0, 3])}`
- 其中两个数组都表示二维坐标

这种设计的优点是可读性强，也便于调试。于是我们可以把观测空间声明为 `Dict`，其中 `agent` 和 `target` 都是表示坐标的 `Box` 空间。

```python
from typing import Optional

import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5):
        # 正方形网格大小，默认 5x5
        self.size = size

        # 初始位置，真正值会在 reset() 时随机生成
        # 这里用 -1, -1 表示“尚未初始化”
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # 定义观测空间
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    0, size - 1, shape=(2,), dtype=int
                ),
                "target": gym.spaces.Box(
                    0, size - 1, shape=(2,), dtype=int
                ),
            }
        )

        # 定义动作空间：4 个方向
        self.action_space = gym.spaces.Discrete(4)

        # 把动作编号映射成实际移动方向
        self._action_to_direction = {
            0: np.array([0, 1]),   # 向右
            1: np.array([-1, 0]),  # 向上
            2: np.array([0, -1]),  # 向左
            3: np.array([1, 0]),   # 向下
        }
```

## 构造观测

由于 `Env.reset()` 和 `Env.step()` 都要返回观测，因此最好写一个辅助方法 `_get_obs()`，把内部状态转换成对外观测格式。这样可以避免重复代码，后续如果改观测格式，也只需要改一个地方。

```python
def _get_obs(self):
    """把内部状态转换成观测格式。"""
    return {
        "agent": self._agent_location,
        "target": self._target_location,
    }
```

类似地，也可以写一个 `_get_info()` 用来返回附加信息。这个教程里，`info` 里放的是智能体和目标之间的曼哈顿距离，适合调试和观察进度，但通常不应该直接作为学习算法的输入。

```python
def _get_info(self):
    """计算调试信息。"""
    return {
        "distance": np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
    }
```

有些场景中，`info` 还会包含只在 `step()` 中才知道的数据，比如奖励拆分项、动作是否成功等。这种情况下，可以在 `step()` 里对 `_get_info()` 的结果继续补充。

## `reset()` 函数

`reset()` 用于开始一个新的 episode。它有两个可选参数：

- `seed`：用于可复现随机结果
- `options`：用于传递额外配置

最重要的一点是：在 `reset()` 的第一行必须调用 `super().reset(seed=seed)`，这样 Gymnasium 才能正确初始化随机数生成器。

在这个 `GridWorld` 里，`reset()` 的工作是：

- 随机生成智能体位置
- 随机生成目标位置
- 确保两者初始位置不相同
- 返回初始观测和 `info`

```python
def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    """开始新的 episode。"""
    # 很重要：必须先调用它，才能正确设置随机种子
    super().reset(seed=seed)

    # 随机生成智能体初始位置
    self._agent_location = self.np_random.integers(
        0, self.size, size=2, dtype=int
    )

    # 随机生成目标位置，且不能与智能体重合
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
        self._target_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )

    observation = self._get_obs()
    info = self._get_info()
    return observation, info
```

## `step()` 函数

`step()` 是环境逻辑的核心。它接收一个动作，更新环境状态，并返回一步交互的结果。物理规则、游戏规则、奖励函数，通常都在这里实现。

对于这个 `GridWorld`，`step()` 主要要做这些事：

1. 把离散动作映射成移动方向
2. 更新智能体位置，并处理边界
3. 计算奖励
4. 判断 episode 是否结束
5. 返回观测、奖励和状态信息

```python
def step(self, action):
    """执行一个时间步。"""
    # 将动作编号映射为移动方向
    direction = self._action_to_direction[action]

    # 更新位置，并保证不会走出边界
    self._agent_location = np.clip(
        self._agent_location + direction, 0, self.size - 1
    )

    # 判断是否到达目标
    terminated = np.array_equal(
        self._agent_location, self._target_location
    )

    # 这里不使用截断逻辑
    truncated = False

    # 简单奖励：到达目标给 1，否则给 0
    reward = 1 if terminated else 0

    observation = self._get_obs()
    info = self._get_info()
    return observation, reward, terminated, truncated, info
```

原文也提到一种常见改法：如果你想鼓励智能体“更快到达目标”，可以给每一步一个很小的负奖励，而不是单纯只在终点给奖励。

## 常见环境设计陷阱

## 奖励设计问题

问题：只在最后成功时给奖励，也就是稀疏奖励，可能让学习非常困难。

```python
# 稀疏奖励，学习可能较难
reward = 1 if terminated else 0
```

更好的做法之一，是提供更有引导性的中间反馈：

```python
# 方案 1：每一步加小惩罚，鼓励更快到达
reward = 1 if terminated else -0.01

# 方案 2：基于距离做 reward shaping
distance = np.linalg.norm(self._agent_location - self._target_location)
reward = 1 if terminated else -0.1 * distance
```

## 状态表示问题

问题：要么塞进太多无关信息，要么缺少关键状态信息。

```python
# 过多信息：智能体并不需要每步都知道网格大小
obs = {
    "agent": self._agent_location,
    "target": self._target_location,
    "size": self.size,
}

# 信息不足：只给距离不够，无法分辨具体位置
obs = {"distance": distance}
```

更合理的方式是只提供做最优决策所需的信息：

```python
obs = {
    "agent": self._agent_location,
    "target": self._target_location,
}
```

## 动作空间问题

问题：动作定义和环境能力不匹配。

```python
# 不合理：动作空间允许斜向移动，但环境逻辑并不支持
self.action_space = gym.spaces.Discrete(8)

# 不合理：明明是离散移动，却用了连续动作空间
self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
```

动作空间应该准确反映环境真正支持的控制方式。

## 边界处理问题

问题：没有处理越界，或越界后的语义不清晰。

```python
# 错误：智能体可能走出网格
self._agent_location = self._agent_location + direction
```

或者即使检查了越界，也没有定义清楚行为：

```python
if np.any(self._agent_location < 0) or np.any(self._agent_location >= self.size):
    # 是保持原地、结束 episode，还是额外惩罚？不明确
    pass
```

更好的方式是明确而一致地处理边界，例如直接裁剪到合法范围：

```python
self._agent_location = np.clip(
    self._agent_location + direction, 0, self.size - 1
)
```

## 注册并创建环境

你可以直接实例化自定义环境，但更常见、更方便的方式，是把它注册到 Gymnasium 里。这样后续就能像内置环境一样通过 `gymnasium.make()` 创建。

环境 ID 一般由三部分组成：

- 可选命名空间
- 必填名称
- 可选但推荐的版本号

原文示例里使用的是完整格式：`gymnasium_env/GridWorld-v0`。

由于这里的教程不是一个正式 Python 包，所以 `entry_point` 直接传类对象。在真实项目里，更常见的是写成字符串，例如 `"my_package.envs:GridWorldEnv"`。

```python
import gymnasium as gym

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,
)
```

注册之后，就可以像内置环境一样创建：

```python
import gymnasium as gym

# 创建环境
env = gym.make("gymnasium_env/GridWorld-v0")

# 传递自定义参数
env = gym.make("gymnasium_env/GridWorld-v0", size=10)
print(env.unwrapped.size)  # 10

# 创建向量化环境
vec_env = gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
```

你也可以通过 `gymnasium.pprint_registry()` 查看已注册环境列表。

## 调试你的环境

自定义环境第一次写出来时，常见问题很多。官方给了几种很实用的调试方式。

## 检查环境有效性

Gymnasium 自带 `check_env`，可以帮你发现很多常见错误。

```python
from gymnasium.utils.env_checker import check_env

try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
```

## 用已知动作手工测试

手工构造一组固定动作，尤其配合固定随机种子，是验证环境行为的高效方法。

```python
env = gym.make("gymnasium_env/GridWorld-v0")
obs, info = env.reset(seed=42)
print(f"起始位置 - Agent: {obs['agent']}, Target: {obs['target']}")

actions = [0, 1, 2, 3]  # 右、上、左、下

for action in actions:
    old_pos = obs["agent"].copy()
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs["agent"]
    print(f"动作 {action}: {old_pos} -> {new_pos}, reward={reward}")
```

## 常见调试错误

### 忘记调用 `super().reset()`

```python
def reset(self, seed=None, options=None):
    # super().reset(seed=seed)  # 缺少这一行
    pass
```

这通常会导致随机种子行为不正确。

### 坐标约定混乱

原文特别提醒了一点：如果你脑子里用的是笛卡尔坐标 `[x, y]`，而实现时又混进了 NumPy 常见的 `[row, col]` 语义，在渲染阶段很容易出现“方向看起来不对”的错觉。

```python
self._action_to_direction = {
    0: np.array([1, 0]),  # 本来想表示“向右”，实际上变成改行号
    1: np.array([0, 1]),  # 本来想表示“向上”，实际上变成改列号
}
```

### 没有处理边界

```python
self._agent_location = self._agent_location + direction
```

这种写法会直接允许智能体跑出网格。

## 使用 Wrappers

有时候你希望修改环境行为，但又不想改核心实现，这时就适合使用 `Wrapper`。它可以在不动原始环境代码的前提下：

- 修改观测格式
- 增加时间限制
- 改写奖励
- 叠加额外功能

示例中用了 `FlattenObservation`，把字典观测压平成一维数组：

```python
from gymnasium.wrappers import FlattenObservation

env = gym.make("gymnasium_env/GridWorld-v0")
print(env.observation_space)

obs, info = env.reset()
print(obs)

wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space)

obs, info = wrapped_env.reset()
print(obs)  # [agent_x, agent_y, target_x, target_y]
```

当你使用的算法只接受固定维度的一维向量输入时，这种包装器会很有用。

## 进阶环境特性

## 添加渲染

如果你希望环境能可视化，可以实现 `render()`：

```python
def render(self):
    """以文本方式渲染环境。"""
    if self.render_mode == "human":
        for y in range(self.size - 1, -1, -1):
            row = ""
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row += "A "
                elif np.array_equal([x, y], self._target_location):
                    row += "T "
                else:
                    row += ". "
            print(row)
        print()
```

这只是最简单的 ASCII 渲染，但已经足够做基础调试。

## 参数化环境

环境也可以设计成支持多个参数，例如：

```python
def __init__(
    self,
    size: int = 5,
    reward_scale: float = 1.0,
    step_penalty: float = 0.0,
):
    self.size = size
    self.reward_scale = reward_scale
    self.step_penalty = step_penalty
```

然后在 `step()` 中按参数动态计算奖励：

```python
if terminated:
    reward = self.reward_scale
else:
    reward = -self.step_penalty
```

这样同一个环境类就可以衍生出多个不同难度或不同奖励风格的任务。

## 真实环境设计建议

## 从简单开始，逐步加复杂度

- 第一步：先把基本移动和到达目标做对
- 第二步：再增加障碍、多目标、时间压力
- 第三步：最后再引入复杂动力学、部分可观测或多智能体交互

## 为“可学习性”设计

- 成功标准要清晰，让智能体知道什么是好表现
- 难度要合理，不能过于简单，也不能难到几乎学不出来
- 规则要一致，同样状态下的同样动作应有同样结果
- 观测要充分，至少包含做出最优决策所必需的信息

## 围绕研究问题设计

- 导航类任务：重点考虑空间推理与路径规划
- 控制类任务：重点考虑动力学、稳定性和连续动作
- 策略类任务：重点考虑部分信息、对手建模和长期规划
- 优化类任务：重点考虑清晰的权衡关系和资源约束

## 下一步

掌握这篇之后，可以继续尝试：

- 给环境添加渲染
- 在你的自定义环境上训练智能体
- 尝试不同奖励函数，观察它们对学习效果的影响
- 组合使用不同包装器来调整环境行为
- 创建更复杂的环境，例如加入障碍、多智能体或连续动作

这篇教程最核心的经验是：先做一个简单、清晰、可验证的环境，然后不断测试、不断迭代，再逐步增加复杂度。
