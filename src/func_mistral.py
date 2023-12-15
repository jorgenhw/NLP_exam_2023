from sentence_transformers import SentenceTransformer # semantic similarity
from sklearn.metrics.pairwise import cosine_similarity # cosine similarity
import pandas as pd
from tqdm import tqdm # loading bar
import regex as re # regex
from danlp.models import load_bert_tone_model


from ctransformers import AutoModelForCausalLM # using mistral model
from pathlib import Path # path stuff

# For arguments
from argparse import ArgumentParser


# load data
def load_data(path: str) -> pd.DataFrame:
    """
    path: path to the data, must be a csv file
    """
    df = pd.read_csv(path)
    return df


# Loading LLM model
def load_llm(temperature: float = 0.8, 
             top_p: float = 0.95, 
             top_k: int = 40, 
             max_new_tokens: int = 1000, 
             context_length: int = 6000,
             repetition_penalty: float = 1.1):

    model_dir = Path.cwd() / 'model' / 'openhermes-2.5-mistral-7b.Q4_K_M.gguf'
    #model_name_gguf = 'openhermes-2.5-mistral-7b.Q4_K_M.gguf'
    #model_name_gptq = 'Mistral-7B-Instruct-v0.2-DARE-GPTQ'

    # Check if the gguf model exists
    #if os.path.isfile(model_dir / model_name_gguf):
    #    model_name = model_name_gguf
    #    # model path
    #    model_path = model_dir / model_name
    # If not, check if the gptq model exists
    #elif os.path.isfile(model_dir / model_name_gptq):
    #    model_name = model_name_gptq
        # model path
    #    model_path = model_dir / model_name
    #    print(model_path)
    #else:
    #    raise Exception("No valid model found in the model directory")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        model_type="mistral",
        gpu_layers=50,
        temperature=temperature, # default is 0.8
        top_p = top_p,
        top_k = top_k,  # default is 40
        max_new_tokens = max_new_tokens,
        context_length = context_length,
        repetition_penalty=repetition_penalty)
    
    #print(f"Using loaded model: {model_name}")
    
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


def generate_internal_dataframe(df, original_column_name: str, paraphrases: list):
    # original_column_name: the name of the column in df containing the original text
    # paraphrases: the paraphrased texts
    
    # Create a DataFrame for original texts and rename the original_column to "Original"
    original_df = df[[original_column_name]].rename(columns={original_column_name: 'Original'})
    
    # Create a DataFrame for paraphrases with column name "New"
    paraphrase_df = pd.DataFrame({'New': paraphrases})
    
    # Combine the original and paraphrase DataFrames by index, where index alignment is assumed
    combined_df = pd.concat([original_df, paraphrase_df], axis=1)
    
    # Optionally, re-attach other columns from the original df that you want to keep besides the original text
    combined_df = pd.concat([df.drop(columns=[original_column_name]), combined_df], axis=1)

    # save combinedf_df as csv in data folder
    combined_df.to_csv(Path.cwd() / 'data' / 'combined_df.csv', index=False)

    return combined_df


# generate semantic similarity scores and append them as a column to the dataframe
def semantic_similarity(df, model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    df: dataframe with original and paraphrased text
    model: the model to use for semantic similarity, default is the multilingual MiniLM model
    """
    model = SentenceTransformer(model)
    similarities = []
    
    for row in tqdm(df.iterrows()):
        embeddings = model.encode([row[1]['Original'], row[1]['New']])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity)
    df['semantic_similarity'] = similarities
    return df


def paraphrase_clean_func(df: pd.DataFrame, 
                          org_col_name: str, 
                          para_col_name: str, 
                          min_length: int = 0, 
                          max_length: int = 500, 
                          min_semantic_similarity: 
                          float = 0.5, 
                          max_semantic_similarity: float = 0.95) -> pd.DataFrame:
    """
    This function takes a dataframe with columns containing original text and 
    paraphrased text and returns a dataframe with a new column containing the 
    paraphrased text without the additional information that is added by the
    paraphrase generator. It also removes paraphrases that are too long or too
    short and paraphrases that are too similar to the original text. Length and
    similarity can be adjusted by the user. 
    """

    df[para_col_name] = df[para_col_name].apply(lambda x: re.sub("(?s)<.im_end.>.*", "", x)) # specific for using Mistral

    # calculate length of paraphrase
    df['new_length'] = df[para_col_name].apply(lambda x: len(x))

    # remove rows above a certain length AND outside the range of semantic similarity
    df = df[(df['new_length'] > min_length) & (df['new_length'] < max_length) & (df['semantic_similarity'] > min_semantic_similarity) & (df['semantic_similarity'] < max_semantic_similarity)].copy()

    # remove rows with a large change in sentiment
    classifier = load_bert_tone_model()

    # add columns with probabilities
    df['probabilities_text'] = df[org_col_name].apply(lambda x: classifier.predict_proba(x, analytic=False)[0])
    df['probabilities_para'] = df[para_col_name].apply(lambda x: classifier.predict_proba(x, analytic=False)[0])

    # difference betweeen highest propability in org and the corresponding probability in new
    df['probabilities_para_argmax'] = df['probabilities_text'].apply(lambda x: x.argmax())
    df['selected_probability'] = df.apply(lambda row: row['probabilities_para'][int(row['probabilities_para_argmax'])], axis=1)
    df['diff_org_new'] = df['probabilities_text'].apply(lambda x: max(x)) - df['selected_probability']

    df = df[(df['diff_org_new'] > -0.3) & (df['diff_org_new'] < 0.3)].copy()

    # remove columns that were added for calculating length and similarity
    df.drop(columns=['new_length', 'probabilities_text', 'probabilities_para', 'probabilities_para_argmax', 'selected_probability', 'diff_org_new', 'semantic_similarity', org_col_name], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


##########################################################
#            DATAFRAME FOR FINAL PRINTOUT                #
##########################################################
def generate_final_dataframe(original_df: pd.DataFrame, new_df: pd.DataFrame, original_col_name: str) -> pd.DataFrame:
    """
    Combines an original dataframe with a new dataframe resulting from the paraphrase_clean_func,
    adds a label indicating whether the text is original or new,
    and returns the combined dataframe.
    
    Parameters:
        original_df (pd.DataFrame): Dataframe with the original texts.
        new_df (pd.DataFrame): Dataframe with the paraphrased texts.
        original_col_name (str): The name of the column in the original dataframe that contains the text to paraphrase.
        
    Returns:
        pd.DataFrame: The combined dataframe with labels.
    """
    
    # Step 1 & 2: Modify original dataframe
    modified_original_df = original_df.copy()
    modified_original_df['org_or_new'] = 1
    modified_original_df.rename(columns={original_col_name: 'New'}, inplace=True)
    
    # Step 3: Combine the dataframes with new labels
    # Assigning label=0 for new_df here to avoid modifying the original data
    combined_df = pd.concat([new_df.assign(org_or_new=0), modified_original_df], ignore_index=True)

    # save combined_df as csv in data folder
    combined_df.to_csv(Path.cwd() / 'data' / 'combined_df_15dec.csv', index=False)

    return combined_df