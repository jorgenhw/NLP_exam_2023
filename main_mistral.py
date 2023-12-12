# before running this, run the following in the terminal: 
#1# run setup.sh
#2# CT_METAL=1 pip install ctransformers --no-binary ctransformers
#3# huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
import pandas as pd
# playing with mistral
from src.func_mistral import load_llm, generate_dataframe, semantic_similarity, generate_paraphrases, load_data
from pathlib import Path
from src.classes import bcolors


def main():
    print(f'{bcolors.HEADER}Running main_mistral.py{bcolors.ENDC}')
 
    print("--------------------------------------------------")

    print(f"{bcolors.BOLD}Before continuing, make sure you have run the following in the terminal (if not, end this and do so, making sure you're in the right working directory (see README)): {bcolors.ENDC}")
    print("")
    print("CT_METAL=1 pip install ctransformers --no-binary ctransformers")
    print("")
    print("huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False")
    print("")
    input(f"{bcolors.OKCYAN}If you have done the above, press Enter to continue...{bcolors.ENDC}")
    print("--------------------------------------------------")
    print(f"{bcolors.BOLD}Put your .csv data file in the 'data' folder.{bcolors.ENDC}")
    print("")
    input(f"{bcolors.OKCYAN}If you have done the so, press Enter to continue...{bcolors.ENDC}")
    print("")
    print(f"{bcolors.BOLD}Insert the name of the csv file you want to paraphrase (without the .csv extension): {bcolors.ENDC}")
    print("")
    filename = input()

    # load CSV
    print("Loading data...")
    train_data = load_data(Path.cwd() / 'data' / f'{filename}.csv')
    print("Data loaded")
    print("")
    print("loading model...")
    # load model
    model = load_llm(Path.cwd() / 'model' / 'openhermes-2.5-mistral-7b.Q4_K_M.gguf')
    print("Model loaded")
    print("")
    
    print("--------------------------------------------------")
    print("Generating paraphrases...")
    # generate paraphrases
    paraphrasings = generate_paraphrases(train_data, 'text', model)
    print("Paraphrases generated")
    print("")

    print("--------------------------------------------------")
    print("Generating dataframe...")
    # generate dataframe with original and paraphrased text and save it as a csv in the data folder
    df = generate_dataframe(train_data, 'text', paraphrasings)
    print("Dataframe generated")
    print("")

    # generate semantic similarity scores and append them as a column to the dataframe
    semantic_similarity(df)

    
if __name__ == "__main__":
    main()


