import pandas as pd
import re

def paraphrase_clean_func(df: pd.DataFrame, org_col_name: str, para_col_name: str, new_col_name: str, min_length: int, max_length: int, min_semantic_similarity: float = 0.5, max_semantic_similarity: float = 0.95):
    """
    This function takes a dataframe with a column named 'paraphrase' and
    returns a dataframe with a column named 'paraphrase_clean' that has
    the paraphrase column cleaned of any non-ascii characters, and
    lowercased.
    """

    df[new_col_name] = df[para_col_name].apply(lambda x: re.sub("(?s)<.im_end.>.*", "", x)) # specific for using Mistral
    
    # calculate length of paraphrase
    df['old_length'] = df[org_col_name].apply(lambda x: len(x))

    # calculate length of paraphrase
    df['new_length'] = df[new_col_name].apply(lambda x: len(x))

    # remove rows above a certain length AND outside the range of semantic similarity
    df = df[(df['new_length'] > min_length) & (df['new_length'] < max_length) & (df['semantic_similarity'] > min_semantic_similarity) & (df['semantic_similarity'] < max_semantic_similarity)]
    
    return df



