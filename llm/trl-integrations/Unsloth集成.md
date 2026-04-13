# Unsloth 集成

> 说明：本文为 Hugging Face TRL 文档 `Unsloth Integration` 的中文翻译版。
> 原文链接：[https://huggingface.co/docs/trl/unsloth_integration](https://huggingface.co/docs/trl/unsloth_integration)
> 文档来源：Hugging Face TRL Documentation
> 访问日期：2026-04-13

Unsloth 是一个开源的微调与强化学习框架，可用于训练大语言模型（例如 Llama、OpenAI gpt-oss、Mistral、Gemma、DeepSeek 等），训练速度最高可提升到 2 倍，同时最多可减少 80% 的显存占用。Unsloth 还支持与 `llama.cpp`、Ollama、vLLM 等其他推理引擎配合完成训练、评估、运行和部署。

该库提供了一套精简且兼容 Hugging Face 的工作流，覆盖训练、评估、推理和部署，并且与 [`SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) 完全兼容。

## 关键特性

- 支持所有兼容 Transformer 的模型训练，包括文本转语音（TTS）、多模态、BERT、强化学习等
- 支持全量微调、预训练、LoRA、QLoRA、8-bit 训练等多种方式
- 可运行在 Linux、Windows、Colab、Kaggle 上，支持 NVIDIA GPU，并计划支持 AMD 与 Intel 环境
- 兼容 TRL 支持的大多数能力，包括 RLHF（如 GSPO、GRPO、DPO 等）
- 通过手写 Triton Kernel 和手动反向传播引擎保证无精度退化（0% 近似误差）

## 安装

### pip 安装

本地安装（官方建议优先在 Linux 上使用）：

```bash
pip install unsloth
```

你也可以按照 [官方文档](https://docs.unsloth.ai) 的说明安装 `unsloth`。安装完成后，将 unsloth 集成到现有流程中非常简单；相比加载 [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/main_classes/model#transformers.AutoModelForCausalLM)，你只需要改为加载 `FastLanguageModel`：

```python
import torch
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

max_length = 2048  # 支持自动 RoPE Scaling，因此可以自行设置长度

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b",
    max_seq_length=max_length,
    dtype="auto",  # 自动检测。Tesla T4/V100 通常用 Float16，Ampere+ 通常用 Bfloat16
    load_in_4bit=True,  # 使用 4bit 量化降低显存占用，也可以设为 False
)

# 对模型打补丁并添加高速 LoRA 权重
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # 当前对 dropout=0 做了优化
    bias="none",  # 当前对 bias="none" 做了优化
    use_gradient_checkpointing=True,
    random_state=3407,
)

training_args = SFTConfig(output_dir="./output", max_length=max_length)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

保存后的模型与 Hugging Face 的 `transformers` 库完全兼容。更多内容可参考 Unsloth 的[官方仓库](https://github.com/unslothai/unsloth)。

### Docker 安装

```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

随后访问 `http://localhost:8888` 打开 Jupyter Lab，即可开始微调。

## 训练

在训练前，可以先调整以下几个核心设置：

- `max_seq_length = 2048`：控制上下文长度。虽然 Llama-3 支持 `8192`，但官方建议测试时先用 `2048`。Unsloth 支持更长上下文的微调，最长可到 4 倍。
- `dtype = "auto"`：自动检测数据类型；在较新的 GPU 上，也可以显式使用 `torch.float16` 或 `torch.bfloat16`。
- `load_in_4bit = True`：启用 4-bit 量化，可将微调时的内存占用降低约 4 倍。如果关闭该选项，则可以启用 LoRA 的 16-bit 微调。
- 若要启用全量微调（FFT），设置 `full_finetuning = True`。若要启用 8-bit 微调，设置 `load_in_8bit = True`。注意：同一时间只能有一种训练方式设为 `True`。

关于 Unsloth 超参数和特性的更多配置方式，可参考其[文档指南](https://docs.unsloth.ai)。

## 保存模型

Unsloth 允许你直接将微调后的模型保存为一个较小的 LoRA Adapter 文件。如果你希望把模型上传出去，也可以直接推送到 Hugging Face Hub。记得提前准备好 [Hugging Face Token](https://huggingface.co/settings/tokens) 并在推送时配置它。

### 保存为 GGUF

保存为 GGUF 时，Unsloth 底层使用的是 `llama.cpp`。如果要保存到本地，可使用：

```python
model.save_pretrained_gguf("directory", tokenizer, quantization_method="q4_k_m")
model.save_pretrained_gguf("directory", tokenizer, quantization_method="q8_0")
model.save_pretrained_gguf("directory", tokenizer, quantization_method="f16")
```

如果要推送到 Hugging Face Hub，可使用：

```python
model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method="q4_k_m")
model.push_to_hub_gguf("hf_username/directory", tokenizer, quantization_method="q8_0")
```

### 保存为 vLLM 可用格式

如果你需要保存为适用于 vLLM 的 16-bit 模型，可使用：

```python
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")
```
