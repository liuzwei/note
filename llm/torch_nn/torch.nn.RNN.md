# torch.nn.RNN

官方文档：<https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html>

`torch.nn.RNN` 用来实现多层 Elman RNN。默认非线性函数是 `tanh`，也可以切换成 `relu`。

如果把一个序列按时间步写成 `x_t`，那么每一层在时刻 `t` 都会结合：

- 当前输入 `x_t`
- 上一个时刻的隐藏状态 `h_(t-1)`

来得到新的隐藏状态 `h_t`。

## 1. 构造函数

```python
torch.nn.RNN(
    input_size,
    hidden_size,
    num_layers=1,
    nonlinearity="tanh",
    bias=True,
    batch_first=False,
    dropout=0.0,
    bidirectional=False,
    device=None,
    dtype=None,
)
```

## 2. 参数含义

- `input_size`：每个时间步输入特征的维度。
- `hidden_size`：隐藏状态 `h` 的维度。
- `num_layers`：RNN 层数。大于 1 时，上一层的输出会作为下一层的输入。
- `nonlinearity`：可选 `"tanh"` 或 `"relu"`，默认是 `"tanh"`。
- `bias`：是否启用 `bias_ih` 和 `bias_hh`。
- `batch_first`：如果为 `True`，输入输出张量按 `(batch, seq, feature)` 排列；否则按 `(seq, batch, feature)` 排列。
- `dropout`：当 `num_layers > 1` 时，会在各层之间加 Dropout，但最后一层之后不会加。
- `bidirectional`：是否使用双向 RNN。若为 `True`，方向数 `D=2`，否则 `D=1`。
- `device`、`dtype`：张量设备和数据类型。

## 3. 输入与输出形状

先约定：

- `N`：batch size
- `L`：sequence length
- `H_in`：`input_size`
- `H_out`：`hidden_size`
- `D`：方向数，单向时为 `1`，双向时为 `2`

### 输入

`input` 支持三种形状：

- 非 batch 输入：`(L, H_in)`
- `batch_first=False`：`(L, N, H_in)`
- `batch_first=True`：`(N, L, H_in)`

`hx` 是初始隐藏状态：

- 非 batch 输入：`(D * num_layers, H_out)`
- batch 输入：`(D * num_layers, N, H_out)`

如果不传 `hx`，会默认用全 0 初始化。

另外，`input` 也可以是变长序列打包后的 `PackedSequence`。

### 输出

`output` 表示最后一层在每个时间步的输出：

- 非 batch 输入：`(L, D * H_out)`
- `batch_first=False`：`(L, N, D * H_out)`
- `batch_first=True`：`(N, L, D * H_out)`

`h_n` 表示最后一个时间步的隐藏状态：

- 非 batch 输入：`(D * num_layers, H_out)`
- batch 输入：`(D * num_layers, N, H_out)`

需要特别注意：

- `batch_first` 只影响 `input` 和 `output` 的排列方式。
- `hx` 和 `h_n` 的维度顺序不受 `batch_first` 影响。
- 对非 batch 输入，`batch_first` 会被忽略。

## 4. 关键理解

### 单层 RNN

单层时，每个时间步都会根据当前输入和上一步隐藏状态计算新的隐藏状态，再把这个隐藏状态传到下一个时间步。

### 多层 RNN

多层时：

- 第 1 层读入原始输入序列
- 第 2 层读入第 1 层在各时间步的输出
- 更高层依次类推

所以 `num_layers=2` 可以理解成“把两个 RNN 堆起来”。

### 双向 RNN

双向时会同时做：

- 正向序列建模
- 反向序列建模

最终输出最后一维会变成 `2 * hidden_size`。

如果你想把双向输出拆开看，可以按官方文档里的思路 reshape：

```python
output = output.view(seq_len, batch, num_directions, hidden_size)
```

其中正向和反向方向分别对应方向 `0` 和 `1`。

## 5. 可学习参数

对第 `k` 层，常见参数包括：

- `weight_ih_l[k]`：输入到隐藏状态的权重。
- `weight_hh_l[k]`：隐藏状态到隐藏状态的权重。
- `bias_ih_l[k]`：输入侧偏置。
- `bias_hh_l[k]`：隐藏状态侧偏置。

其中：

- 第 0 层的 `weight_ih_l[k]` 形状是 `(hidden_size, input_size)`
- 更高层的 `weight_ih_l[k]` 形状是 `(hidden_size, num_directions * hidden_size)`

参数初始化来自均匀分布，范围与 `hidden_size` 相关。

## 6. 一个最小示例

```python
import torch
from torch import nn

rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=False,
)

input = torch.randn(5, 3, 10)   # (seq_len=5, batch=3, input_size=10)
h0 = torch.randn(2, 3, 20)      # (num_layers=2, batch=3, hidden_size=20)

output, hn = rnn(input, h0)

print(output.shape)  # (5, 3, 20)
print(hn.shape)      # (2, 3, 20)
```

如果改成双向：

```python
rnn = nn.RNN(10, 20, num_layers=2, bidirectional=True)
input = torch.randn(5, 3, 10)
h0 = torch.randn(4, 3, 20)  # 4 = 2 directions * 2 layers

output, hn = rnn(input, h0)

print(output.shape)  # (5, 3, 40)
print(hn.shape)      # (4, 3, 20)
```

## 7. 使用时容易混淆的点

- `output` 是“最后一层在每个时间步的输出”，不是所有层的输出全集。
- `h_n` 是“最后一个时间步的隐藏状态”，并且会保留所有层、所有方向。
- `batch_first=True` 不会改变 `h0` / `h_n` 的维度顺序。
- `dropout` 不是每个时间步后都加，而是加在层与层之间，且最后一层后不加。
- 输入如果是 `PackedSequence`，输出也会是 `PackedSequence`。

## 8. 官方文档里的额外说明

- 某些 `cuDNN` / `CUDA` 版本上，RNN 计算可能存在非确定性行为。
- 如果你要求严格可复现，官方文档建议设置相关环境变量。
- 在特定 GPU、dtype 和输入条件下，PyTorch 可能选择持久化算法来提升性能。

## 9. 一句话总结

`torch.nn.RNN` 是最基础的循环神经网络层：它按时间步迭代处理序列，用隐藏状态传递历史信息，并支持多层、双向和变长序列输入。
