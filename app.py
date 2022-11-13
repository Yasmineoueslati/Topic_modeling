import time
import streamlit as st
import pandas as pd
import numpy as np

from functionforDownloadButtons import download_button
import os
import json

from utils import topics_suggestion, positive_review

st.set_page_config(
    page_title="Topic Modeling",
    page_icon="🎈",
    layout="wide",
)

df = pd.read_csv('data/dataset_negative.csv', sep=",", index_col=None)
df_cleaned = pd.read_csv('data/dataset_cleaned.csv', sep=",", index_col=None)
df_cleaned.columns = ["Texte", "Stars", "Length", "Cleaned Text"]


def index_input_callback():
    st.session_state['options'] = df.iloc[index_input]['text']


def aleatoire_callback():
    random_index = np.random.randint(df.shape[0], size=1)[0]
    st.session_state['index_input'] = random_index
    st.session_state['options'] = df.iloc[index_input]['text']


def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

with st.sidebar:
    ModelType = st.radio(
        "Choose between the two options",
        ["Avis dataset", "Texte libre"],
    )
    if ModelType == "Avis dataset":
        index_input = st.number_input("Write index number", key="index_input", step=1, min_value=0,
                                      max_value=df.shape[0],
                                      on_change=index_input_callback)
        st.button("random", on_click=aleatoire_callback)

    number = st.slider('Choose the number of topics', value=3, step=1, min_value=1, max_value=15)

    with open('style.css') as f:
        css_component = f'<style>{f.read()}</style>'
    st.markdown(css_component, unsafe_allow_html=True)

st.markdown("<h3> Topic Modeling App </h3>", unsafe_allow_html=True)
with st.expander("🤔 About Topic Modeling"):
    st.write(
        """ Topic modeling is a broad term. It encompasses a number of specific statistical learning methods. 
        These methods do the following: explain documents in terms of a set of topics and those topics in terms of 
        the a set of words. Two very commonly used methods are Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF), 
        for instance. Used without additional qualifiers the approach is usually assumed to be unsupervised 
        although there are semi-supervised and supervised variants.""")
    from PIL import Image

    image = Image.open('wordcloud.png')
    st.image(image, caption='Most frequent words')
    st.markdown("")
with st.expander("💻️ About this project"):
    st.write(
        """ The intention of this project is to develop and implement text pre-processing skills and feature extraction techniques 
        specific to text-type unstructured data in order to detect dissatisfaction subjects mentioned by customers in their 
        opinions posted on the customer reviews platform. The project covers the entire cycle of implementation of 
        concept proof, from the pre-treatment of data to the deployment""")
    st.markdown("")
with st.expander(" 💡 How to use this app "):
    video_file = open('streamlit-topicModeling.webm', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
with st.expander("🎯 Topic Moodeling Steps"):
    etape_1, etape_2, etape_3, etape_4 = st.columns(4)
    etape_1.info("❶ Data cleaning and pre-treatment")
    etape_2.info("❷ Data vectorization and modeling")
    etape_3.info("❸ Web application development")
    etape_4.info("❹ Web application deployment")
with st.expander("📥 Download source code and the dataset"):
    col1, col2, col3, col4 = st.columns([2.2, 2.9, 3.7, 4])


    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(df)
    col1.download_button(
        label="Download data",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',
    )
    code_file1 = open('app.py', 'r', encoding="utf8").read()
    col2.download_button(
        label="Download app.py code",
        data=code_file1
    )
    st.code(code_file1, language='python')

    code_file2 = open('machine_learning/preprocessing.py', 'r', encoding="utf8").read()
    col3.download_button(
        label="Download preprocessing code",
        data=code_file2
    )
    code_file3 = open('machine_learning/build_model.py', 'r', encoding="utf8").read()
    col4.download_button(
        label="Download modeling code",
        data=code_file3
    )






st.markdown("<h3> Enter your text below</h3>", unsafe_allow_html=True)
review = st.text_area("✍ Write the opinion", height=200, max_chars=5000, key='options')
detect_topic_btn = st.button(label="✨ Detect the source of dissatisfaction")

if detect_topic_btn:
    test = positive_review(review)
    if test:
        st.warning("✔ Your opinion is positive, write negative opinion to detect the topic")
    else:
        suggested_topics = topics_suggestion(review, number)
        columns_components = st.columns(len(suggested_topics))
        i = 0
        list1 = []
        list2 = []
        for col in columns_components:
            col.metric(suggested_topics[i][0], suggested_topics[i][1])
            list1.append(suggested_topics[i][0])
            list2.append(float(suggested_topics[i][1].replace("%", "")))
            i += 1
        st.balloons()

        "Probability per topic"
        source = pd.DataFrame({
            'Probabilité': list2,
            'Topic': list1
        })
        import altair as alt

        bar_chart = alt.Chart(source).mark_bar(color="#DE3163").encode(
            y='Probabilité:Q',
            x='Topic:O',
        )
        st.altair_chart(bar_chart, use_container_width=True)

        if len(suggested_topics) != number:
            st.warning(
                "Le nombre de topic que vous avez demandé est supérieur au nombre de topic "
                "qui peuvent être en relation avec ce review (Probabilité de similarité égale à 0%)"
            )

format_dictionary = {
    "Relevancy": "{:.1%}",
}
