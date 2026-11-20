"""
Main module for poet-brain.

A model trainer.
"""

from statistics import mode
from unicodedata import block

from accelerate.test_utils.scripts.test_notebook import NUM_PROCESSES
from datasets import load_dataset
from jedi.third_party.typeshed.stubs.openpyxl.openpyxl.styles.builtins import total
from transformers import AutoTokenizer

poems = load_dataset("text", data_files="data/poems/*.txt", sample_by="document", split="train")
poems = poems.train_test_split(test_size=0.05, seed=0)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])


tokenized_poems = poems.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=poems["train"].column_names,
)

block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_lenght = len(concatenated_examples[list(examples.keys())[0]])

    if total_lenght >= block_size:
        total_lenght = (total_lenght // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_lenght, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_poems.map(group_texts, batched=True, num_proc=4)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

training_args = TrainingArguments(
    output_dir="out/poetry-brain",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()

import math

evel_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
