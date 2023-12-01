import openai
import os
from dotenv import load_dotenv # load .env file
from sentence_transformers import SentenceTransformer # semantic similarity
from sklearn.metrics.pairwise import cosine_similarity # cosine similarity
import pandas as pd
from tqdm import tqdm # loading bar

from ctransformers import AutoModelForCausalLM # using mistral model
from pathlib import Path # path stuff



# Set the OpenAI API key
def set_api_key():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Paraphrase a list of strings and output a list of paraphrased strings (using gpt-3)
def paraphrase_text_list(text_list):
    paraphrased_list = []
    print("Paraphrasing...")
    for text in tqdm(text_list):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Paraphrase the following sentence (in Danish): "},
                {"role": "user", "content": text},
            ]
        )
        paraphrased_text = response['choices'][0]['message']['content']
        paraphrased_list.append(paraphrased_text)
    return paraphrased_list

# Creates a dataset with original text and paraphrased text
def df_with_original_and_paraphrased_text(original_text_list, paraphrased_text_list):
    df = pd.DataFrame({
        'original_text': original_text_list, 
        'paraphrased_text': paraphrased_text_list
    })
    return df

def make_input_mistral(phrase: str) -> str:
    system = f"""Du er en sprogmodel som forstår og taler kompetent dansk.
    Du svarer kun på dansk.
    Din opgave er at omformulere tekst ved at finde synonymer med samme betydning.
    Her er nogle eksempler på omformuleringer:
    {examples.original[0]} -> {examples.paraphrased[0]},
    {examples.original[1]} -> {examples.paraphrased[1]}
    Her er nogle regler du skal overholde:
    Du må ikke gentage dig selv.
    Du må ikke bruge samme sætning som i originalen.
    Du skal bruge samme kontekst som i originalen.
    """

    prompt = f"""
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {phrase}<|im_end|>
    <|im_start|>assistant
    """

    return prompt


# Creates a similarty metric between original and paraphrased sentences in the df ^ created above
def semantic_similarity(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    similarities = []
    for index, row in df.iterrows():
        embeddings = model.encode([row['original_text'], row['paraphrased_text']])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity)
    df['semantic_similarity'] = similarities
    return df


