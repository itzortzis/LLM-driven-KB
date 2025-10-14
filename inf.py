from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("tinyllama-finetuned", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")


# prompt = """### Instruction:
# From the given input produce some 5 tags
# ### Input:
# A short article about ingredients, restaurants, and desserts in the context of food.
# ### Response:"""

prompt = """### Instruction:
From the given input, extract key information and represent it as RDF triples (subject, predicate, object).
### Input:
Ioannis Tzortzis is working as developer in Athens. he lives there as well.
### Response:"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
