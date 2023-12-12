#####################
#     Libraries     #
#####################

import openai
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Set the OpenAI API key
def set_api_key():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


###################################
#    PARAPHRASING W. CHAT GPT     #
###################################

# Paraphrase a list of strings and output a list of paraphrased strings (using gpt-3):
def paraphrase_text_list(text_list: list) -> list:
    """
    Takes a list of strings and paraphrases them using the gpt-3 chat model.
    
    text_list: list of strings to paraphrase
    """
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
