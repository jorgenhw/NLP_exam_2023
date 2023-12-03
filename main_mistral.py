# before running this, run the following in the terminal: 
#1# run setup.sh
#2# CT_METAL=1 pip install ctransformers --no-binary ctransformers
#3# huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
import pandas as pd
# playing with mistral
from src.func import load_mistral, generate_dataframe
from pathlib import Path

# load csv
train_data = pd.read_csv(Path.cwd() / 'data' / 'train_data.csv')


# Load example data and data to paraphrase (in this case the same data, shouldn't be...)
examples = pd.read_csv(Path.cwd() / 'data' / 'examples.csv')

# loads model -> takes data to paraphrase as input and example paraphrasings as input -> returns dataframe with original and paraphrased text
df = generate_dataframe(train_data.iloc[:,1], 
                        load_mistral(Path.cwd() / 'model' / 'openhermes-2.5-mistral-7b.Q4_K_M.gguf'),
                        'paraphrasings.csv')