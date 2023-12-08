# before running this, run the following in the terminal: 
#1# run setup.sh
#2# CT_METAL=1 pip install ctransformers --no-binary ctransformers
#3# huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
import pandas as pd
# playing with mistral
from src.func_mistral import load_llm, generate_dataframe, semantic_similarity, generate_paraphrases
from pathlib import Path
from src.classes import bcolors


def main():
    print(f'{bcolors.HEADER}Running main_mistral.py{bcolors.ENDC}')
    
    print("--------------------------------------------------")
    print("--------------------------------------------------")

    print("Before continuing, make sure you have run the following in the terminal (if not, end this and do so, making sure you're in the right working directory (see README)): ")
    print("")
    print("CT_METAL=1 pip install ctransformers --no-binary ctransformers")
    print("")
    print("huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False")
    print("")
    input("If you have done the above, press Enter to continue...")
    print("--------------------------------------------------")
    
    # load CSV
    print("Loading data...")
    train_data = pd.read_csv(Path.cwd() / 'data' / 'train_data.csv')

    # load model
    model = load_llm(Path.cwd() / 'model' / 'openhermes-2.5-mistral-7b.Q4_K_M.gguf')

    # generate paraphrases
    paraphrasings = generate_paraphrases(train_data, 'text', model)

    # generate dataframe with original and paraphrased text and save it as a csv in the data folder
    df = generate_dataframe(train_data, 'text', paraphrasings)

    # generate semantic similarity scores and append them as a column to the dataframe
    semantic_similarity(df)

    
if __name__ == "__main__":
    main()



# Ask the user for their choice of options
def process_user_choice():
    while True:
        print("#############################################")
        print(f"{bcolors.OKCYAN}Choose an option (1 / 2):{bcolors.ENDC}")
        print("#############################################")
        print("1. Enter your own URL to an article from The Guardian website")
        print("2. Get the latest The Guardian Briefing")
        
        choice = input("Enter the number of your choice: ")

        if choice == '1':
            url = ask_for_url()
            return url
        elif choice == '2':
            url = get_most_recent_briefing()
            return url
        else:
            print(f"{bcolors.WARNING}Invalid choice. Please choose option 1 or 2.{bcolors.ENDC}")