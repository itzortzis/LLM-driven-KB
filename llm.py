from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback
from datasets import load_dataset
import numpy as np
import argparse
import evaluate
import os
import torch
import gc

parser = argparse.ArgumentParser(description="Add integers from command line.")

parser.add_argument("c", type=int, help="Chunk size")
parser.add_argument("i", type=int, help="Chunk index")
parser.add_argument("e", type=int, help="Epochs")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


class VRAMCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print("\n--- Running VRAM cleanup before evaluation ---")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        gc.collect()

        
        print("-------------------------------------------\n")


model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


dataset = load_dataset("GEM/web_nlg", "en")
# print(dataset)
# train_set = dataset["train"].select(range(10))
val_chunk = int(0.2 * args.c / 0.8)
print(val_chunk)
valid_set = dataset["validation"].select(range(val_chunk))



train_set_all = dataset["train"].shuffle(seed=42)

start = args.i * args.c
end = start + args.c
# print(start, end)
train_set = train_set_all.select(range(start, end))



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



def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # If preds are logits (which they are for CausalLM), take argmax
    preds = np.argmax(preds, axis=-1)

    # Replace ignored positions (-100) with pad_token_id
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode text
    with torch.no_grad():
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
    # print(example)
    instruction = "From the given input produce a reasonable RDF triple"
    text = f"### Instruction:\n{instruction}\n### Input:\n{example['target']}\n### Response:\n{example['input']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_tokenized = train_set.map(tokenize_fn)
valid_tokenized = valid_set.map(tokenize_fn)
print(train_tokenized)
print(valid_tokenized)

trained_model_name = "tinyllama-finetuned_" + str(args.c) + "_" + str(args.i)
run_name = "LLM-driven-KB-" + str(args.c) + "_" + str(args.i)

args = TrainingArguments(
    output_dir="./tinyllama-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=args.e,
    fp16=True,
    save_steps=500,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="wandb",
    run_name=run_name
)

trainer = Trainer(model=model, args=args, train_dataset=train_tokenized, eval_dataset=valid_tokenized, compute_metrics=compute_metrics, callbacks=[VRAMCleanupCallback()])
trainer.train()
model.save_pretrained(trained_model_name)
