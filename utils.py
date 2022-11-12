import numpy as np
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sklearn

uploaded_pickled_model = pickle.load(open('models/model_yasmine', 'rb'))
model_vectorizer = pickle.load(open('models/vectorizer_yasmine', 'rb'))
topics = ['👨‍Staff management', '🍔 Food Quality', '🍕 Pizza', '🍗 Menu Chicken', '👌 Quality', '🕑 Service time',
           '🍔 Burger', '🕑 Waiting Time', '💼 Experience', '🍹 Drinks', '📦 Ordering & Delivery to table', '🗺️ Location',
           '💁 Customer Service',  '🍣 Sushi and Rice', '🏘️ Place Environnement']


def negative_review(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] <= 0 :
        return False
    return True


def topics_suggestion(text, nb):
    transformed_text = model_vectorizer.transform([text])
    predicted_topics = uploaded_pickled_model.transform(transformed_text)
    sorted_predicted_topics = np.argsort(predicted_topics, axis=1)
    final_predicted_topics = []
    for i in range(len(predicted_topics)):
        for j in range(len(topics) - 1, len(topics) - 1 - nb, -1):
            topic_index = sorted_predicted_topics[i][j]
            topic = topics[topic_index]
            topic_percentage = round(100*predicted_topics[i][topic_index], 1)
            if topic_percentage == 0:
                break
            final_predicted_topics.append([topic, str(topic_percentage)+"%"])
    return final_predicted_topics
