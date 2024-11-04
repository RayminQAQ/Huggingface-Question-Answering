import os
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
    AutoConfig,
    DataCollatorForSeq2Seq,
)
from datasets import DatasetDict
from util import Data

# Parameters
BASE_MODEL_PATH = "path/to/your/base/model"  # Add your model path here
OUTPUT_DIR = "result"

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs', 
    logging_steps=20,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
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

########## TBD
# Tokenizer and Model Configuration
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
config = AutoConfig.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL_PATH)

# Data Collator for Causal Language Modeling
# Ref
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    #source_max_len=512,  # Adjust max lengths based on requirements
    #target_max_len=128,
    train_on_source=True,
    predict_with_generate=False,
)

# Evaluation metric

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start Training
trainer.train()

print(trainer.state.log_history)