from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Model and dataset
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dataset_path = "customer_support.jsonl"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
dataset = load_dataset("json", data_files=dataset_path, split="train")

def format(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding='max_length', max_length=512)

dataset = dataset.map(format)

# Load base model and apply LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="llama3-lora-ollama",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# Save LoRA adapter and tokenizer
model.save_pretrained("llama3-lora-ollama")
tokenizer.save_pretrained("llama3-lora-ollama")
