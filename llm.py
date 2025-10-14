from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# dataset = load_dataset("json", data_files="dataset.json")

ds = load_dataset("GEM/web_nlg", "en", split="train[:5000]")
dataset = ds


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize_fn(example):
    instruction = "From the given input produce a reasonable RDF triple"
    text = f"### Instruction:\n{instruction}\n### Input:\n{example['target']}\n### Response:\n{example['input']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn)
print(tokenized)

args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    optim="paged_adamw_8bit",
    report_to="wandb"
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
model.save_pretrained("tinyllama-finetuned")
