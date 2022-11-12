import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions

#import en_core_web_sm
#nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


# Tokenization du texte
tokenizer = RegexpTokenizer(r'\w+')
def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed


# Lémmatisation du texte
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    # print(tokens_tagged)
    lemmatized_text_list = list()
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,
                                                             'a'))  # Lemmatise adjectives. Not doing anything since we remove all adjective
            # print(word, 'a')
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'v'))  # Lemmatise verbs
            # print(word, 'v')
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'n'))  # Lemmatise nouns
            # print(word, 'n')
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word, 'r'))  # Lemmatise adverbs
            # print(word, 'r')
        else:
            lemmatized_text_list.append(
                lemmatizer.lemmatize(word))  # If no tags has been found, perform a non specific lemmatisation
    return " ".join(lemmatized_text_list)


# Normalisation du texte
def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])


# Suppression des contractions
def contraction_text(text):
    return contractions.fix(text)


# Création des tokens négatifs
negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"
def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i + 1 for i in range(len(tokens) - 1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx] = negative_prefix + tokens[idx]
    tokens = [token for i, token in enumerate(tokens) if i + 1 not in negative_idx]
    return " ".join(tokens)


# Suppression des mots courants (stopwords)
from spacy.lang.en.stop_words import STOP_WORDS
def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]

    return " ".join([word for word in text.split() if word not in english_stopwords])


# Pré-traitement final du jeu de données
def preprocess_text(text):
    # Tokenize review
    text = tokenize_text(text)
    # Lemmatize review
    text = lemmatize_text(text)
    # Normalize review
    text = normalize_text(text)
    # Remove contractions
    text = contraction_text(text)
    # Get negative tokens
    text = get_negative_token(text)
    # Remove stopwords
    text = remove_stopwords(text)
    return text


DATASET_FILE_PATH = "../data/dataset.csv"
dataset_df = pd.read_csv(DATASET_FILE_PATH)
dataset_df["length"] = dataset_df["text"].apply(lambda x: len(x.split()))

# Enregistrer Cleaned Dataset
dataset_df["text_cleaned"] = dataset_df["text"].apply(preprocess_text)
dataset_df.to_csv("../data/dataset_cleaned.csv", index=False)

# Extraction du jeu de données d'avis négatifs
dataset_negative_df = dataset_df[dataset_df.stars < 3]
dataset_negative_df.to_csv("../data/dataset_negative.csv", index=False)
