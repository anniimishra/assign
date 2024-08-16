from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
model_name = "meta-llama/Llama-2-7b-hf"  
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_text = "Once upon a time, in a land far away,"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: {input_text}\n")
print(f"Generated Text: {generated_text}")