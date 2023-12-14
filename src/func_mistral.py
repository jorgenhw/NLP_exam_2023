from sentence_transformers import SentenceTransformer # semantic similarity
from sklearn.metrics.pairwise import cosine_similarity # cosine similarity
import pandas as pd
from tqdm import tqdm # loading bar

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
def load_llm(model_path: str, temperature: float = 0.8, max_new_tokens: int = 1000, context_length: int = 6000):
    """
    model_path: path to the model
    """
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), 
        model_type="mistral",
        gpu_layers=50,
        temperature=temperature, # default is 0.8
        top_p = 0.95,
        top_k = 40,  # default is 0.95
        max_new_tokens = max_new_tokens,
        context_length = context_length)
    
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

##########################################################
#            DATAFRAME FOR INTERNAL FILTERING OF DF      #
##########################################################

# generate a dataframe with the original and paraphrased text
def generate_internal_dataframe(df, original_column_name: list, paraphrases: list): # original and paraphrases = list of strings
    
     """
     original: the original text to paraphrase
     paraphrases: the paraphrased texts
     """
     original = df[original_column_name].tolist()
     data = list(zip(original, paraphrases))
     df = pd.DataFrame(data, columns=['Original', 'New']) # create dataframe with original and paraphrased text
     
     return df


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
    # save df as csv
    df.to_csv(Path.cwd() / 'data' / 'paraphrasings_w_semantics.csv', index=False)
    return None


def paraphrase_clean_func(df: pd.DataFrame, org_col_name: str, para_col_name: str, min_length: int, max_length: int, min_semantic_similarity: float = 0.5, max_semantic_similarity: float = 0.95, sentiment_method = 'danlp') -> pd.DataFrame:
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
    df['old_length'] = df[org_col_name].apply(lambda x: len(x))

    # calculate length of paraphrase
    df['new_length'] = df[para_col_name].apply(lambda x: len(x))

    # remove rows above a certain length AND outside the range of semantic similarity
    df = df[(df['new_length'] > min_length) & (df['new_length'] < max_length) & (df['semantic_similarity'] > min_semantic_similarity) & (df['semantic_similarity'] < max_semantic_similarity)]

    # remove rows with a large change in sentiment
    if sentiment_method == 'danlp':
        from danlp.models import load_bert_tone_model

        classifier = load_bert_tone_model()

        # add columns with probabilities
        df['probabilities_text'] = df['text'].apply(lambda x: classifier.predict_proba(x, analytic=False)[0])
        df[['prob_pos_org', 'prob_neu_org', 'prob_neg_org']] = pd.DataFrame(df.probabilities_text.tolist(), index= df.index)
        df['probabilities_para'] = df['text_paraphrase_clean'].apply(lambda x: classifier.predict_proba(x, analytic=False)[0])
        df[['prob_pos_new', 'prob_neu_new', 'prob_neg_new']] = pd.DataFrame(df.probabilities_para.tolist(), index= df.index)
    
        # difference betweeen highest propability in org and the corresponding probability in new
        df['probabilities_para_argmax'] = df['probabilities_text'].apply(lambda x: x.argmax())
        df['selected_probability'] = df.apply(lambda row: row['probabilities_para'][int(row['probabilities_para_argmax'])], axis=1)
        df['diff_org_new'] = df['probabilities_text'].apply(lambda x: max(x)) - df['selected_probability']

        df = df[(df['diff_org_new'] > -0.3) & (df['diff_org_new'] < 0.3)].copy()
    
    # remove columns that were added for calculating length and similarity
    df.drop(columns=['old_length', 'new_length', 'probabilities_text', 'prob_pos_org', 'prob_neu_org', 'prob_neg_org', 'probabilities_para', 'prob_pos_new', 'prob_neu_new', 'prob_neg_new', 'probabilities_para_argmax', 'selected_probability', 'diff_org_new', 'text_paraphrase', 'semantic_similarity', 'text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df



##########################################################
#            DATAFRAME FOR FINAL PRINTOUT         #
##########################################################

def generate_dataframe(df, original_column_name: str, paraphrases: list):
    # original_column_name: the name of the column in df containing the original text
    # paraphrases: the paraphrased texts
    
    # Create a DataFrame for original texts and another for paraphrased texts
    original_df = df.copy()
    paraphrase_df = df.copy()
    
    # Add the paraphrased texts to the paraphrase DataFrame
    paraphrase_df[original_column_name] = paraphrases
    
    # Add a column to indicate if the sentence is original (1) or paraphrased (0)
    original_df['is_original'] = 1
    paraphrase_df['is_original'] = 0
    
    # Combine the original and paraphrase DataFrames
    combined_df = pd.concat([original_df, paraphrase_df], ignore_index=True)
    
    # Keep all the original columns except the text column, which is now merged with paraphrases
    combined_columns = [col for col in df.columns if col != original_column_name] + ['is_original', original_column_name]
    combined_df = combined_df[combined_columns]
    
    # Rename the original text column if needed
    combined_df.rename(columns={original_column_name: 'Text'}, inplace=True)
    
    # Save the combined DataFrame as CSV
    combined_df.to_csv(Path.cwd() / 'data' / 'combined_original_paraphrased.csv', index=False)
    
    return combined_df