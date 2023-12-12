import src.func_gpt as func

# Import openai key
func.set_api_key()

# Test the paraphrase_text function
text_to_paraphrase = ["Danmark kan ikke leve af gæld.", 
                      "Så det går da trods alt, også i forhold til tallene fra sidste år, den rigtige vej.",
                      "Det er en kæmpe lettelse, at vi nu kan åbne op for rejser til en række lande uden for Europa."]

# getting the paraphrased text
paraphrase = func.paraphrase_text_list(text_to_paraphrase)


# Test the df_with_original_and_paraphrased_text function
df = func.df_with_original_and_paraphrased_text(text_to_paraphrase, paraphrase)

# semantic similarity
similarity = func.semantic_similarity(df)

# print the df
print(df)