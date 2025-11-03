from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("checkpoints/checkpoint-38", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("checkpoints/checkpoint-38", local_files_only=True)

prompt = "What is the meaning of people being alive?"
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tok.decode(outputs[0]))
