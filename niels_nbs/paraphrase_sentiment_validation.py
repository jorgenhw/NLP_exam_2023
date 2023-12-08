from danlp.models import load_bert_tone_model
classifier = load_bert_tone_model()

# using the classifier
classifier.predict('Analysen viser, at økonomien bliver forfærdelig dårlig')
'''{'analytic': 'objektive', 'polarity': 'negative'}''' 
classifier.predict('Jeg tror alligvel, det bliver godt')
'''{'analytic': 'subjektive', 'polarity': 'positive'}'''

# Get probabilities and matching classes names
classifier.predict_proba('Analysen viser, at økonomien bliver forfærdelig dårlig')
classifier._classes()