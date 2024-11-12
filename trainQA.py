import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoConfig, GenerationConfig
)
import evaluate
from datasets import DatasetDict
from util import Data
import numpy as np

# Parameters
BASE_MODEL_PATH = "fnlp/bart-base-chinese"  # Add your model path here
DATA_PATH = "Data"
OUTPUT_DIR = "result"

# Tokenizer and Model Configuration
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
config = AutoConfig.from_pretrained(BASE_MODEL_PATH)

model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH, config=config)

model.config.decoder_start_token_id = model.config.decoder_start_token_id or tokenizer.bos_token_id or tokenizer.cls_token_id
model.config.bos_token_id = model.config.bos_token_id or tokenizer.bos_token_id or tokenizer.cls_token_id
model.config.eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id or tokenizer.sep_token_id

# Define generation arguments
generation_config = GenerationConfig(
    max_length=128,            
    min_length=20,            
    do_sample=True,            
    num_beams=5,              
    top_k=50,                  
    top_p=0.9,                
    temperature=0.7,           
    early_stopping=True,       
    repetition_penalty=1.2,    # 避免重複生成相同的短語
    length_penalty=1.0,         # 長度懲罰，較高值傾向於生成更長的句子
    
    # 设置特殊标记 ID
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id,
    eos_token_id=tokenizer.eos_token_id or tokenizer.sep_token_id,
)


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    seed=42,
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    remove_unused_columns=True,
    predict_with_generate=True,
    generation_config=generation_config,
    
    eval_strategy="steps",  # Evaluation strategy
    save_strategy="steps",  # Save strategy
    report_to="none",  # Reporting strategy
    
    logging_steps=500,  # Logging steps
    save_total_limit=2,  # Limit the total number of saved models
    load_best_model_at_end=True,  # Load the best model at the end
)

# Load data
data_loader = Data(path_to_dir=DATA_PATH)
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
print("Loaded DatasetDict: ", dataset_dict)


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
metric = evaluate.load("rouge")  

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # 將 predictions 中的 -100 替換為 pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 解碼預測和標籤
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 簡單的後處理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
   
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
