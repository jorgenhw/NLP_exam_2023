import pandas as pd
from paraphrase_clean_funcs import paraphrase_clean_func

# reading data
df_para = pd.read_csv('data/twitter_data_paraphrasings_w_semantics.csv', index_col=False)
df_train = pd.read_csv('data/twitter_data_train.csv', index_col=False)

# rename columns
df_para.rename(columns={'New': 'text_paraphrase'}, inplace=True)
df_para.rename(columns={'Original': 'text'}, inplace=True)

# combine data
df_para['label'] = df_train['label']

df = paraphrase_clean_func(df_para, org_col_name='text', para_col_name='text_paraphrase', new_col_name = 'text_paraphrase_clean', min_length = 0, max_length = 500, min_semantic_similarity = 0.7, max_semantic_similarity = 0.95)

# write to csv
df.to_csv('data/twitter_data_paraphrasings_cleaned.csv', index=False)