from datasets import Dataset
import pandas as pd
import os


# Load and preprocess the dataset
def load_and_prepare_dataset(file_name):

    # Load the dataset
    dataset = pd.read_csv(os.path.join('data', f'{file_name}.csv'))

    # make dataset only 500 long 
    dataset = dataset[:500] 

    # Remove all rows where language is not 'da'
    dataset = dataset[dataset['language'] == 'da']

    # Remove all columns except 'text' and 'label'
    dataset = dataset[['text', 'label']]

    # Remove all duplicates
    dataset = dataset.drop_duplicates()

    # Convert to dict and then to a Hugging Face Dataset
    dataset = Dataset.from_dict(dataset)

    print("Dataset loaded and prepared")

    return dataset

# Split the dataset and convert into a Hugging Face DatasetDict
from datasets import DatasetDict

def split_dataset(dataset, seed=42):
    # 60% train, 20% validation, 20% test
    train_test = dataset.train_test_split(test_size=0.4, seed=seed) 
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=seed)

    new_train = pd.read_csv(os.path.join("data", "combined_df_10DEC.csv"))
    new_train = new_train[['text', 'label']]
    new_train = Dataset.from_dict(new_train)
    # changing the train dataset to the new train dataset
    
    

    # combine train, test and valid to one dictionary
    dataset_splitted_dict = DatasetDict({
        'train': train_test['train'],
        'valid': test_valid['train'],
        'test': test_valid['test']})
    # change the train dataset to the new train dataset
    dataset_splitted_dict['train'] = new_train
    print("changing train dataset to new train dataset")
    print("successfully changed train dataset to new train dataset")
    print("printing shape: ")
    print(dataset_splitted_dict)
    print("-------------------------")
    
    # make a pandas dataframe of of the train dataset
    #df_train = dataset_splitted_dict['train'].to_pandas()
    #df_train.to_csv(os.path.join("data", "TO_PARAPHRASE_9DEC.csv"))
    
    print("Dataset splitted into train (60%), valid (20%) and test (20%)")

    # output the train dataset as a csv file
    #dataset_splitted_dict['train'].to_csv(os.path.join("data", "train.csv"))

    return dataset_splitted_dict

# Tokenize the dataset 
from transformers import AutoTokenizer
from datasets import ClassLabel
from src.classes import *


def tokenize_dataset(dataset, model_name="NbAiLab/nb-bert-large", max_length=128):
    # defining the labels
    labels_cl = ClassLabel(num_classes=3, names=['negative', 'neutral', 'positive'])

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # defining a function to tokenize the text and translate all labels into integers instead of strings
    def tokenize_function(example):
        tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
        tokens['label'] = labels_cl.str2int(example['label'])
        return tokens

    # actually tokenizing the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names) # batched=True speeds up tokenization by allowing to process multiple lines at once

    print(f"{bcolors.BOLD}Dataset tokenized{bcolors.ENDC}")

    return tokenized_dataset

# evaluation metrics
import numpy as np
import evaluate

def compute_metrics(eval_pred):
    metric0 = evaluate.load("accuracy")
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric0.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric1.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Training arguments
from transformers import TrainingArguments

def training_arguments(batch_size=16, epochs=5, learning_rate=2e-4):
    training_args = TrainingArguments(output_dir="test_trainer",
                                      num_train_epochs=epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      warmup_steps=100,
                                      weight_decay=0.01,
                                      logging_dir="logs",
                                      logging_steps=10,
                                      load_best_model_at_end=True,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",  # Add this line
                                      remove_unused_columns=False,
                                      run_name="test_trainer")
    return training_args



# Training
from transformers import TrainingArguments, Trainer

def training(model, training_args, tokenized_dataset, compute_metrics):

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

# Get classiciation report and metrics
from sklearn.metrics import classification_report
import os

def get_classification_report(trainer, tokenized_dataset):
    predictions, labels, metrics = trainer.predict(tokenized_dataset['valid'])
    predictions = np.argmax(predictions, axis=1)
    print(classification_report(labels, predictions, target_names=['negative', 'neutral', 'positive']))
    pd.DataFrame(metrics).to_csv(os.path.join("classification_reports", "metrics_TEST.csv"))
    print(metrics)