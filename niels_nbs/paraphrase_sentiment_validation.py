from danlp.models import load_bert_tone_model
import pandas as pd

classifier = load_bert_tone_model()

df = pd.read_csv('data/twitter_data_paraphrasings_cleaned.csv', index_col=False)

# choose a subset of the data
df = df.sample(n=10, random_state=1)

# add columns with probabilities
df['probabilities_text'] = df['text'].apply(lambda x: classifier.predict_proba(x, analytic=False)[0])
df[['prob_pos_org', 'prob_neu_org', 'prob_neg_org']] = pd.DataFrame(df.probabilities_text.tolist(), index= df.index)
df['probabilities_para'] = df['text_paraphrase_clean'].apply(lambda x: classifier.predict_proba(x, analytic=False)[0])
df[['prob_pos_new', 'prob_neu_new', 'prob_neg_new']] = pd.DataFrame(df.probabilities_para.tolist(), index= df.index)
#classifier._classes()