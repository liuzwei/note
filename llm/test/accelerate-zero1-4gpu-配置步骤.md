# Qwen3-8B 全参微调 Accelerate 配置步骤

4张RTX PRO 6000  96G

适用脚本：[qwen3-8b-sft-full-deepspeed.py](C:/develop/code/note/llm/test/qwen3-8b-sft-full-deepspeed.py)

## 1. 生成配置

执行：

```bash
accelerate config
```

按下面的顺序选择：

```text
In which compute environment are you running?
This machine

Which type of machine are you using?
multi-GPU

How many different machines will you use (use more than 1 for multi-node training)? [1]:
1

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:
NO

Do you wish to optimize your script with torch dynamo? [yes/NO]:
NO

Do you want to use DeepSpeed? [yes/NO]:
yes

Do you want to specify a json file to a DeepSpeed config? [yes/NO]:
NO

What should be your DeepSpeed's ZeRO optimization stage?
1

How many gradient accumulation steps you're passing in your script? [1]:
8

Do you want to use gradient clipping? [yes/NO]:
yes

What is the gradient clipping value? [1.0]:
1.0

Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
NO

Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]:
NO

How many GPU(s) should be used for distributed training? [1]:
4

Do you wish to use mixed precision?
bf16
```

配置会保存到：

```text
/root/.cache/huggingface/accelerate/default_config.yaml
```

## 2. 生成后的配置内容

对应的 `default_config.yaml` 内容如下：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 8
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

说明：

- `zero_stage: 1` 表示使用 DeepSpeed ZeRO-1。
- `num_processes: 4` 表示使用 4 张卡。
- `mixed_precision: bf16` 表示使用 BF16。
- `gradient_accumulation_steps: 8` 要和训练脚本里的设置保持一致。

## 3. 启动训练

执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file /root/.cache/huggingface/accelerate/default_config.yaml \
  /root/sft-full-deepspeed.py
```

如果你是在当前目录这份脚本上训练，可以对应改成：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file /root/.cache/huggingface/accelerate/default_config.yaml \
  /root/llm/test/qwen3-8b-sft-full-deepspeed.py
```

## 4. 开始训练前检查

建议先执行：

```bash
accelerate env
accelerate test --config_file /root/.cache/huggingface/accelerate/default_config.yaml
```

如果脚本里已经写了：

```python
per_device_train_batch_size=8
gradient_accumulation_steps=8
bf16=True
```

那么当前这套配置的等效总 batch size 是：

```text
8 x 8 x 4 = 256
```

这个 batch 对全参训练偏大。如果实际训练出现显存不足，优先把脚本里的 `per_device_train_batch_size` 从 `8` 降到 `1` 或 `2`。
