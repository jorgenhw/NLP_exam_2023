from src.finetuning_a_bert import load_and_prepare_dataset, split_dataset, tokenize_dataset, training, training_arguments, compute_metrics, get_classification_report
import os

path = os.path.join("data", "tweets_data_temp.csv")

def main():
    print("--------------------------------------------------")
    print("Starting finetuning script...")
    print("")
    print("1. Put your correctly formated .csv file (see README) in the data folder")
    input("If you have done so, press enter to continue...")
    input("Type the filename of the dataset without the extension (.csv) of the dataset you want to use: ")    
    file_name = input()
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(file_name)

    print("Splitting dataset...")
    dataset_splitted_dict = split_dataset(dataset)

    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset_splitted_dict)

    print("Loading model (NbAiLab/nb-bert-large)...")
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("NbAiLab/nb-bert-large", num_labels=3)

    # training arguments
    print("Loading training arguments...")
    training_args = training_arguments()

    print("Training model...")
    trainer = training(model, training_args, tokenized_dataset, compute_metrics)

    print("Getting classification report...")
    get_classification_report(trainer, tokenized_dataset)

if __name__ == "__main__":
    main()