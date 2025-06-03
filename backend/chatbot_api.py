from ai_bot.load_fine_tune import model, tokenizer
import torch

def generate_response(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()
