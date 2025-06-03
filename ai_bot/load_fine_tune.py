from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 1. Load tokenizer
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Load LoRA fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "lora-llama3-customerbot", device_map="auto")
model.eval()

# 4. Inference loop
while True:
    prompt = input("\n💬 User: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("🤖 AI:", response.replace(prompt, "").strip())
