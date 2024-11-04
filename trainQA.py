import os
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer, 
    Seq2SeqTrainer,
    AutoConfig,
    DefaultDataCollator,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
import evaluate
from datasets import DatasetDict
from util import Data

# Parameters
BASE_MODEL_PATH = "fnlp/bart-base-chinese"  # Add your model path here
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
    remove_unused_columns=False,
)


generation_config = GenerationConfig(
    max_length=128,            
    min_length=20,            
    do_sample=True,            
    top_k=50,                  
    top_p=0.9,                
    temperature=0.7,           
    num_beams=5,              
    early_stopping=True,       
    repetition_penalty=1.2,    # 避免重複生成相同的短語
    length_penalty=1.0         # 長度懲罰，較高值傾向於生成更長的句子
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
#config.hidden_dropout_prob = 0.2
model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL_PATH, config=config)
model.generation_config = generation_config

# Data preproccess: Preprocessing function
column_names = dataset_dict["train"].column_names

def preprocess_function(examples):
    inputs = examples['Question']
    targets = examples['Answer']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    
    # Set labels for Answer
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset_dict.map(
    preprocess_function, 
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
    )

########## TBD
# Data Collator for Causal Language Modeling
data_collator = DefaultDataCollator()

# Evaluation metric
metric = evaluate.load("squad")

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=[training_args,generation_config],
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metric,
)

# Start Training
trainer.train()

print(trainer.state.log_history)
