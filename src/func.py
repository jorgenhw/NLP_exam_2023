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

###################################
#            CHAT GPT             #
###################################

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

###################################
#            MISTRAL              #
###################################
def make_input_mistral(phrase: str) -> str: # examples should be a dataset with column 0 being the original text and column 1 being the paraphrased text
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

def load_mistral(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), 
        model_type="mistral",
        gpu_layers=50,
        temperature=0.8, # default is 0.8
        top_p = 0.95,
        top_k = 40,  # default is 0.95
        max_new_tokens = 1000,
        context_length = 6000)
    
    return model

def generate_dataframe(original, model, output_name: str): # original = list of strings, model = output from the load_mistral() function
    data = []
    for string in tqdm(original):
        new = model(make_input_mistral(string))
        data.append([string, new])
    df = pd.DataFrame(data, columns=['Original', 'New'])
    # save df as csv
    df.to_csv(Path.cwd() / 'data' / output_name, index=False)
    return df



# Creates a dataset with original text and paraphrased text
def df_with_original_and_paraphrased_text(original_text_list, paraphrased_text_list):
    df = pd.DataFrame({
        'original_text': original_text_list, 
        'paraphrased_text': paraphrased_text_list
    })
    # save df as csv
    df.to_csv(Path.cwd() / 'data' / 'paraphrasings.csv', index=False)
    return df


# Creates a similarty metric between original and paraphrased sentences in the df ^ created above
def semantic_similarity(df):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    similarities = []
    for index, row in tqdm(df.iterrows()):
        embeddings = model.encode([row['Original'], row['New']])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarities.append(similarity)
    df['semantic_similarity'] = similarities
    return df


