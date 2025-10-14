from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
from transformers import TrainerCallback

import evaluate


class SampleGenerationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        sample = valid_set[0]
        prompt = f"### Instruction:\nFrom the given input produce a reasonable RDF triple\n### Input:\n{sample['target']}\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=50)
        print("\n\n--- Sample Generation ---")
        print("Input:", sample["target"])
        print("Generated:", tokenizer.decode(output[0], skip_special_tokens=True))
        print("Reference:", sample["input"])
        print("--------------------------\n")

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# dataset = load_dataset("json", data_files="dataset.json")

dataset = load_dataset("GEM/web_nlg", "en")
train_set = dataset["train"].select(range(1000))
valid_set = dataset["validation"].select(range(100))


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


def compute_metrics_old(eval_pred):
    preds, labels = eval_pred

    # 🔹 preds are logits → take argmax to get token IDs
    preds = np.argmax(preds, axis=-1)

    # 🔹 Replace ignored index (-100) in labels so decoding works
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 🔹 Decode tokens to strings
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 🔹 Simple metric: exact string match
    matches = [int(p.strip() == l.strip()) for p, l in zip(decoded_preds, decoded_labels)]
    exact_match = np.mean(matches)

    return {"exact_match": exact_match}


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # If preds are logits (which they are for CausalLM), take argmax
    preds = np.argmax(preds, axis=-1)

    # Replace ignored positions (-100) with pad_token_id
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process: strip whitespace
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Compute BLEU
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]

    # Compute ROUGE-L (you can also add rouge1, rouge2 if you like)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    rougeL = rouge_result["rougeL"]

    return {
        "bleu": bleu_score,
        "rougeL": rougeL,
    }


def tokenize_fn(example):
    instruction = "From the given input produce a reasonable RDF triple"
    text = f"### Instruction:\n{instruction}\n### Input:\n{example['target']}\n### Response:\n{example['input']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_tokenized = train_set.map(tokenize_fn)
valid_tokenized = valid_set.map(tokenize_fn)
print(train_tokenized)
print(valid_tokenized)

args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="wandb"
)

trainer = Trainer(model=model, args=args, train_dataset=train_tokenized, eval_dataset=valid_tokenized, compute_metrics=compute_metrics, callbacks=[SampleGenerationCallback()])
trainer.train()
model.save_pretrained("tinyllama-finetuned")
