# before running this, run the following in the terminal: 
#1# run setup.sh
#2# CT_METAL=1 pip install ctransformers --no-binary ctransformers
#3# huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
import pandas as pd
# playing with mistral
from src.func_mistral import load_llm, generate_dataframe, semantic_similarity, generate_paraphrases, load_data, generate_internal_dataframe
from pathlib import Path
from src.classes import bcolors

# For arguments
import argparse


def main(args):
    print(f'{bcolors.HEADER}Running main_mistral.py{bcolors.ENDC}')
 
    print("--------------------------------------------------")

    print(f"{bcolors.BOLD}Before continuing, make sure you have run the following in the terminal (if not, end this and do so, making sure you're in the right working directory (see README step 2)): {bcolors.ENDC}")
    print("")
    print("huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --path ./model --path-use-symlinks False")
    print("")
    input(f"{bcolors.OKCYAN}If you have done the above, press Enter to continue...{bcolors.ENDC}")
    print("--------------------------------------------------")
    print(f"{bcolors.BOLD}Put your correctly formatted .csv (0th column containing the strings to paraphrase) data file in the 'data' folder.{bcolors.ENDC}")
    print("")
    input(f"{bcolors.OKCYAN}If you have done the so, press Enter to continue...{bcolors.ENDC}")
    print("--------------------------------------------------")
    print("")
    print(f"{bcolors.BOLD}Insert the name of the csv file you want to paraphrase (without the .csv extension): {bcolors.ENDC}")
    print("")
    filename = input()
    print("")
    print("--------------------------------------------------")
    # load CSV
    print("Loading data...")
    
    
    train_data = load_data(Path.cwd() / 'data' / f'{filename}.csv')
    
    
    print(f'{bcolors.OKGREEN}data loaded{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")
    print("")
    print("Loading model...")
    
    
    # load model
    model = load_llm(Path.cwd() / 'model' / 'openhermes-2.5-mistral-7b.Q4_K_M.gguf',
                     temperature=args.temperature,
                     max_new_tokens=args.max_new_tokens,
                     context_length=args.context_length)
    
    
    print(f'{bcolors.OKGREEN}Model loaded{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")
    print("")
    print(f"{bcolors.BOLD}Insert the name of the column containing the strings to paraphrase (N.B: case sensitive): {bcolors.ENDC}")
    print("")
    column_name = input() #############
    print("")
    print("--------------------------------------------------") 
    print(f'{bcolors.BOLD}Generating paraphrases...{bcolors.ENDC}')
    print("")
    
    
    # generate paraphrases
    paraphrasings = generate_paraphrases(train_data, column_name, model)
    
    
    print("")

    print(f'{bcolors.OKGREEN}Paraphrases generated{bcolors.ENDC}')
    print("")

    print("--------------------------------------------------")
    print("")
    print("Filtering out bad paraphrases...")

    df_internal = generate_internal_dataframe(train_data, column_name, paraphrasings)

    print("--------------------------------------------------")
    print(f'{bcolors.BOLD}Calculating semantic similarity...{bcolors.ENDC}')

    # generate semantic similarity scores and append them as a column to the dataframe
    semantic_similarity(df_internal)
    
    
    print("Adding semantic similarity scores to dataframe...")


    print("--------------------------------------------------")
    print(f'{bcolors.BOLD}Generating dataframe...{bcolors.ENDC}')
    
    # generate dataframe with original and paraphrased text and save it as a csv in the data folder
    df = generate_dataframe(train_data, column_name, paraphrasings)
    
    print("")
    print(f'{bcolors.OKGREEN}Dataframe generated{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")
    print(f'{bcolors.BOLD}Calculating semantic similarity...{bcolors.ENDC}')

    # generate semantic similarity scores and append them as a column to the dataframe
    semantic_similarity(df)
    
    
    print("Adding semantic similarity scores to dataframe...")
    print("")
    print("--------------------------------------------------")
    print("Script successfully run. Check the 'data' folder for the paraphrased data.")
    
if __name__ == "__main__":
    main()


# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating paraphrases for annotated datasets with Mistral')
    # Optional arguments for logstic regression
    parser.add_argument('--temperature', type=float, default=0.9, help='specify the temperature for Mistral, default is 0.9')

    args = parser.parse_args()
    main(args)