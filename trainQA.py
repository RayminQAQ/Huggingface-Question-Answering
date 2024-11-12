import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoConfig,
)
import evaluate
from datasets import DatasetDict
from util import Data
import numpy as np

# Parameters
BASE_MODEL_PATH = "fnlp/bart-base-chinese"  # Add your model path here
OUTPUT_DIR = "result"

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=5,
    remove_unused_columns=True,
)

# Load data
data_loader = Data(path_to_dir='Data')
hf_dataset = data_loader.load_data()
print("Loaded Dataset:", hf_dataset)

# Split data into training, validation, test sets
train_test_dict = hf_dataset.train_test_split(train_size=0.8, test_size=0.2)
val_test_dict = train_test_dict["test"].train_test_split(train_size=0.5, test_size=0.5)
dataset_dict = DatasetDict({
    "train": train_test_dict["train"],
    "validation": val_test_dict["train"],
    "test": val_test_dict["test"],
})

# Tokenizer and Model Configuration
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH, config=config)

# Data preproccess: Preprocessing function
def preprocess_function(examples):
    inputs = examples['Question']
    targets = examples['Answer']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    
    # Set labels for Answer
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset_dict.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_dict["train"].column_names,
    desc="Running tokenizer on dataset",
)

# Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define compute_metrics function
metric = evaluate.load("rouge")  # 对于生成任务，通常使用ROUGE指标

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # 取平均值
    result = {k: v.mid.fmeasure * 100 for k, v in result.items()}
    return result

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start Training
train_result = trainer.train()

# Save train result
print(trainer.state.log_history)
trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
