# My Portfolio Website
# 🦙 LLaMA 3 8B LoRA Fine-Tuning for Customer Support Chatbot

This project demonstrates how to fine-tune the [Meta-LLaMA-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model using the LoRA (Low-Rank Adaptation) technique with a custom dataset for building a customer support chatbot. The result is a personalized LLM capable of answering domain-specific questions accurately and efficiently.

---

## 📁 Project Structure

```bash
ai_bot/
├── train_lora.py           # Fine-tuning script using PEFT + Transformers
├── load_fine_tune.py       # Script to load and test the fine-tuned model
├── dataset.jsonl           # Custom instruction-style training data
├── output/                 # Directory for LoRA adapter output
└── README.md               # Project documentation
```

---

## 📦 Model Information

* **Base Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
* **Fine-Tune Method**: [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
* **Library**: Hugging Face Transformers, PEFT, Datasets, Accelerate
* **Quantization**: 4-bit (bnb\_4bit) via [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

---

## 📊 Dataset Format

We use an instruction-following dataset for chatbot Q\&A:

```json
{
  "prompt": "Customer: How do I reset my password?\nAssistant:",
  "response": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions."
}
```

Tokenized using the LLaMA 3 tokenizer with padding:

```python
tokenizer.pad_token = tokenizer.eos_token
```

---

## 🧠 LoRA Config Used

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 🧪 Training Details

| Hyperparameter | Value       |
| -------------- | ----------- |
| Model          | LLaMA 3 8B  |
| Batch Size     | 4           |
| Epochs         | 3           |
| Max Length     | 512         |
| Learning Rate  | 2e-4        |
| Quantization   | 4-bit (bnb) |

Train using Hugging Face `Trainer` with `PeftModel`.

---

## 🏁 Usage Guide

### 1. Login to Hugging Face

```bash
huggingface-cli login
```

### 2. Download Base Model

```bash
python download_llama.py
```

### 3. Train the Model

```bash
python train_lora.py
```

### 4. Test the Fine-Tuned Model

```bash
python load_fine_tune.py
```

---

## 🤖 Sample Output

```text
User: How do I change my billing address?
Assistant: You can update your billing address by going to your account settings under the Billing section.
```

---

## 🌍 Hosting & Deployment

You can deploy the fine-tuned adapter using:

* Hugging Face Inference API
* Ollama (if merged with base model)
* Gradio UI

---

## ✅ Checklist for Hugging Face Upload

When uploading to Hugging Face Hub:

* ✅ Add `adapter_config.json` and `adapter_model.safetensors`
* ✅ Include `README.md`, `training_args.bin`, and `special_tokens_map.json`
* ✅ Tag with `llama-3`, `lora`, `fine-tuned`, `customer-support`

---

## 🙌 Credits

Created by **Pyae Sone (Monkey455)** with support from Hugging Face & Meta AI.

Contact: [huggingface.co/Monkey455](https://huggingface.co/Monkey455)

---

## 📝 License

MIT License. Free to use, modify, and share.
