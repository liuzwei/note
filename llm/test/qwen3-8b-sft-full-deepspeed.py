import os
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf'
os.environ.setdefault('TENSORBOARD_LOGGING_DIR', '/root/tf-logs')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

MODEL_NAME = 'Qwen/Qwen3-8B'
OUTPUT_DIR = Path('/root/autodl-tmp/sft/Qwen3-8B/sft-full')
BEST_MODEL_DIR = OUTPUT_DIR / 'best'

data_candidates = [
    Path.cwd() / 'datasets' / 'psychology_data.jsonl',
    Path(__file__).resolve().parent / 'datasets' / 'psychology_data.jsonl',
]
data_file = next((path for path in data_candidates if path.exists()), data_candidates[0])

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id


dataset_dict = load_dataset('json', data_files=str(data_file))
dataset_dict = dataset_dict['train'].train_test_split(test_size=0.1, seed=42)

def map_func(example):
    conversation = example['conversation']
    messages = []
    for item in conversation:
        messages.append({'role': 'user', 'content': item['human']})
        messages.append({'role': 'assistant', 'content': item['assistant']})
    return {'messages': messages}

dataset_dict = dataset_dict.map(
    map_func,
    batched=False,
    remove_columns=['dataset', 'conversation', 'category', 'conversation_id'],
)


# Configure trainer
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    seed=42,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,
    warmup_steps=27,
    gradient_checkpointing=True,
    report_to="tensorboard",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer
)

trainer.train()

trainer.save_model(str(BEST_MODEL_DIR))
tokenizer.save_pretrained(str(BEST_MODEL_DIR))
