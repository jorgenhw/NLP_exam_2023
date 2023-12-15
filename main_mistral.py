# before running this, run the following in the terminal: 
#1# run setup.sh
#2# CT_METAL=1 pip install ctransformers --no-binary ctransformers
#3# huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
import pandas as pd
# playing with mistral
from src.func_mistral import load_llm, semantic_similarity, generate_paraphrases, load_data, generate_internal_dataframe, paraphrase_clean_func, generate_final_dataframe
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
    print(f'{bcolors.BOLD}Loading data...{bcolors.ENDC}')
    
    ##########################################
    train_data = load_data(Path.cwd() / 'data' / f'{filename}.csv')
    ##########################################
    
    print(f'{bcolors.OKGREEN}data loaded{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")
    print("")
    print(f'{bcolors.BOLD}Loading model...{bcolors.ENDC}')
    
    ##########################################
    # load model
    model = load_llm(Path.cwd() / 'model' / 'openhermes-2.5-mistral-7b.Q4_K_M.gguf',
                     temperature=args.temperature,
                     top_k=args.top_k,
                     top_p=args.top_p,
                     max_new_tokens=args.max_new_tokens,
                     context_length=args.context_length)
    ##########################################
    
    print(f'{bcolors.OKGREEN}Model loaded{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")
    print("")
    print(f"{bcolors.BOLD}Insert the name of the column containing the strings to paraphrase (N.B: case sensitive): {bcolors.ENDC}")
    print("")

    ##########################################
    column_name = input() 
    ##########################################
    
    print("")
    print("--------------------------------------------------") 
    print(f'{bcolors.BOLD}Generating paraphrases...{bcolors.ENDC}')
    print("")
    
    ##########################################
    # generate paraphrases
    paraphrasings = generate_paraphrases(train_data, column_name, model)
    ##########################################
    
    print("")

    print(f'{bcolors.OKGREEN}Paraphrases generated{bcolors.ENDC}')
    print("")

    print("--------------------------------------------------")
    print("")
    print(f'{bcolors.BOLD}Filtering out bad paraphrases...{bcolors.ENDC}')
    print("")
    
    ##########################################
    df_internal = generate_internal_dataframe(train_data, column_name, paraphrasings)
    ##########################################

    ##########################################
    # generate semantic similarity scores and append them as a column to the dataframe
    df_w_sem = semantic_similarity(df_internal,
                        model=args.model)
    ##########################################

    ##########################################
    df_int = paraphrase_clean_func(df_w_sem, 'Original', 'New', 
                                   min_length=args.min_length, 
                                   max_length=args.max_length,
                                    min_semantic_similarity=args.min_semantic_similarity,
                                    max_semantic_similarity=args.max_semantic_similarity)
    ##########################################

    print(f'{bcolors.OKGREEN}Bad paraphrases filtered out{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")
    print(f'{bcolors.BOLD}Generating final dataframe...{bcolors.ENDC}')
    
    ##########################################
    # generate dataframe with original and paraphrased text and save it as a csv in the data folder
    generate_final_dataframe(train_data, df_int, column_name)
    ##########################################
    print("")
    print(f'{bcolors.OKGREEN}Final dataframe generated and saved in "data" folder{bcolors.ENDC}')
    print("")
    print("--------------------------------------------------")

    print(f'{bcolors.OKCYAN}Script successfully run. Check the "data" folder for the paraphrased data.{bcolors.ENDC}')
    
# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating paraphrases for annotated datasets with Mistral')
    # Optional arguments for defining the parameters of the mistral model
    parser.add_argument('--temperature', type=float, default=0.8, help='specify the temperature for Mistral, default is 0.9')
    parser.add_argument('--top_p', type=float, default=0.95, help='specify the top_p for Mistral, default is 0.95')
    parser.add_argument('--top_k', type=int, default=40, help='specify the top_k for Mistral, default is 40')
    parser.add_argument('--max_new_tokens', type=int, default=1000, help='specify the max number of new tokens for Mistral, default is 1000')
    parser.add_argument('--context_length', type=int, default=6000, help='specify the context length for Mistral, default is 6000')

    # Change the semantic similarity model
    parser.add_argument('--model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', help='specify the semantic similarity model, default is sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Change the minimum and maximum length of the paraphrases and the minimum and maximum semantic similarity score
    parser.add_argument('--min_length', type=int, default=0, help='specify the minimum length of the paraphrases, default is 0')
    parser.add_argument('--max_length', type=int, default=500, help='specify the maximum length of the paraphrases, default is 500')
    parser.add_argument('--min_semantic_similarity', type=float, default=0.5, help='specify the minimum semantic similarity score, default is 0.5')
    parser.add_argument('--max_semantic_similarity', type=float, default=0.95, help='specify the maximum semantic similarity score, default is 0.95')

    args = parser.parse_args()
    main(args)