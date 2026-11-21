"""
Main module for poet-brain.

A model trainer.
"""

import math
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL = "openai-community/gpt2"
BLOCK_SIZE = 128


def group_texts(examples):
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples}
    total_length = len(concatenated["input_ids"])
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    poems = load_dataset("text", data_files="data/poems/*.txt", sample_by="document", split="train")
    poems = poems.train_test_split(test_size=0.05, seed=0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer([poem + tokenizer.eos_token for poem in examples["text"]])

    tokenized_poems = poems.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=poems["train"].column_names,
    )
    lm_dataset = tokenized_poems.map(group_texts, batched=True, num_proc=4)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    training_args = TrainingArguments(
        output_dir="out/poetry-brain",
        eval_strategy="epoch",
        logging_steps=10,
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

    before = trainer.evaluate()
    print(f"Perplexity before: {math.exp(before['eval_loss']):.2f}")

    trainer.train()
    trainer.save_model("out/poetry-brain/final")

    after = trainer.evaluate()
    print(f"Perplexity after: {math.exp(after['eval_loss']):.2f}")


if __name__ == "__main__":
    main()
