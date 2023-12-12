from sentence_transformers import SentenceTransformer # semantic similarity
from sklearn.metrics.pairwise import cosine_similarity # cosine similarity
import pandas as pd
from tqdm import tqdm # loading bar

from ctransformers import AutoModelForCausalLM # using mistral model
from pathlib import Path # path stuff


# load data
def load_data(path: str) -> pd.DataFrame:
    """
    path: path to the data, must be a csv file
    """
    df = pd.read_csv(path)
    return df

# Loading LLM model
def load_llm(model_path, model_type="mistral"):
    """
    model_path: path to the model
    model_type: the type of model to load, default is mistral
    """
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), 
        model_type=model_type,
        gpu_layers=50,
        temperature=0.8, # default is 0.8
        top_p = 0.95,
        top_k = 40,  # default is 0.95
        max_new_tokens = 1000,
        context_length = 6000)
    
    return model

# Create input for llm model
def make_input_for_llm(phrase: str) -> str: # examples should be a dataset with column 0 being the original text and column 1 being the paraphrased text

    """
    phrase: string to paraphrase
    """

    system = f"""Your task is to proficiently understand and communicate in Danish. You are required to rephrase text in Danish while adhering to the following rules:

1. Avoid repeating yourself.
2. Refrain from using the same sentence as in the original text.
3. Maintain a similar text length to the original.
4. Ensure the context remains consistent with the original text.

Please provide your rephrased response in Danish, observing the given rules and maintaining the context of the original text.
    """

    prompt = f"""
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {phrase}<|im_end|>
    <|im_start|>assistant
    """
    return prompt

# generate paraphrases
def generate_paraphrases(df, column_name, model): 
    """
    df: DataFrame containing the original text
    column_name: name of the column in df containing the original text
    model: the model to use for paraphrasing
    """
    original = df[column_name].tolist()
    paraphrases = []
    for string in tqdm(original):
        paraphrase = model(make_input_for_llm(string))
        paraphrases.append(paraphrase)
    return paraphrases

# generate a dataframe with the original and paraphrased text
def generate_dataframe(df, original_column_name: list, paraphrases: list): # original and paraphrases = list of strings
    
    """
    original: the original text to paraphrase
    paraphrases: the paraphrased texts
    """
    original = df[original_column_name].tolist()
    data = list(zip(original, paraphrases))
    df = pd.DataFrame(data, columns=['Original', 'New']) # create dataframe with original and paraphrased text
    # save df as csv
    df.to_csv(Path.cwd() / 'data' / 'paraphrasingsNEW.csv', index=False)
    return df

# Creates a similarty metric between original and paraphrased sentences in the df ^ created above
def semantic_similarity(df, model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    df: dataframe with original and paraphrased text
    model: the model to use for semantic similarity, default is the multilingual MiniLM model
    """
    
    print("Calculating semantic similarity...")

    print("Loading model...")
    model = SentenceTransformer(model)
    similarities = []
    for row in tqdm(df.iterrows()):
        embeddings = model.encode([row[0], row[1]])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity)
    df['semantic_similarity'] = similarities
    # save df as csv
    df.to_csv(Path.cwd() / 'data' / 'paraphrasings_w_semantics.csv', index=False)
    return df


