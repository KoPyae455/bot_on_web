from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# LoRA adapter folder
adapter_path = "./lora-llama3-customerbot"

# Load PEFT config
config = PeftConfig.from_pretrained(adapter_path)

# Load base model (LLaMA3 8B)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, adapter_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token  # fix pad token issue

# 🔁 Run inference
while True:
    prompt = input("User: ")
    if prompt.lower() == "exit":
        break

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    print("Bot:", tokenizer.decode(outputs[0], skip_special_tokens=True))
