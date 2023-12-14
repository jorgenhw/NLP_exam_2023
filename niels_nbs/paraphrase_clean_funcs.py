import pandas as pd
import re

def paraphrase_clean_func(df: pd.DataFrame, org_col_name: str, para_col_name: str, new_col_name: str, min_length: int, max_length: int, min_semantic_similarity: float = 0.5, max_semantic_similarity: float = 0.95, sentiment_method = 'dacy') -> pd.DataFrame:
    """
    This function takes a dataframe with columns containing original text and 
    paraphrased text and returns a dataframe with a new column containing the 
    paraphrased text without the additional information that is added by the
    paraphrase generator. It also removes paraphrases that are too long or too
    short and paraphrases that are too similar to the original text. Length and
    similarity can be adjusted by the user. 
    """

    df[new_col_name] = df[para_col_name].apply(lambda x: re.sub("(?s)<.im_end.>.*", "", x)) # specific for using Mistral
    
    # calculate length of paraphrase
    df['old_length'] = df[org_col_name].apply(lambda x: len(x))

    # calculate length of paraphrase
    df['new_length'] = df[new_col_name].apply(lambda x: len(x))

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
    df = df.drop(columns=['old_length', 'new_length', 'probabilities_text', 'prob_pos_org', 'prob_neu_org', 'prob_neg_org', 'probabilities_para', 'prob_pos_new', 'prob_neu_new', 'prob_neg_new', 'probabilities_para_argmax', 'selected_probability', 'diff_org_new'])

    return df



