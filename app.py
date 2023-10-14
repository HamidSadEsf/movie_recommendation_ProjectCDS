import streamlit as st
import pandas as pd
import numpy as np

bg_image = f'''
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("Streamlit_Asset\V01_Dark.jpg");
    background-size: cover;
    }}
    </style>
    '''
    
st.markdown(bg_image,  unsafe_allow_html=True)
st.title('Movie Recommendation System App')

pages = ['Introduction', 'Databases', 'Preprocessing', 'Content based', 'Collaborative Filtering', 'Hybrid', 'Conclusion']
page = st.sidebar.radio('Go to', pages)

match page:
    case "Introduction":
        st.write("## Introduction, Title, participants, problem and stakes")
        st.markdown('''![movie recommendation system](/Streamlit_Asset/V01_Dark.jpg)''')
        st.markdown('''I need to highlight these ==very important words==.''')
        st.markdown('''---''')
        st.markdown('''hybrid recommender system is a type of recommender system that combines multiple recommendation techniques or approaches to provide more accurate and effective recommendations. By integrating different methods, hybrid recommender systems aim to overcome the limitations and weaknesses of individual recommendation techniques, resulting in improved recommendation quality. While the details of implementation may differ between various players, such companies as Netflix, Amazon, Spotify, LinkedIn, Youtube and others use hybrid recommenders.''') 
        st.markdown('''This project’s hybrid recommender integrates Content-Based and User-Based Collaborative Filtering Recommenders taking user behavior (history of movies watched) and preferences (explicit rating) into account. This represents a rather straightforward and simple implementation of a hybrid recommender system, which becomes evident when we compare our system's structure to a more sophisticated one, as exemplified by Netflix, for instance.''')
    case "Databases":
        st.write("## presenting the data (volume, architecture), 2/3 data Visualization")

    case "Preprocessing":
        st.write("## data preparation pipeline")

    case "Content based recommendation":
        from model.ContentBasedRec import ContentBasedRecommender
        cbs = ContentBasedRecommender()
        cbs.load_database()
        st.write("## Content based recommendation")
        user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
        level = st.slider("number of recommendations", 1, 20)
        if(st.button('Submit')):
            df_rec_cbs = cbs.recommendation(userId= user, number_of_recommendations= level).drop(['movieId','labels'], axis = 1)
            st.dataframe(df_rec_cbs)
    case "Collaborative Filtering":
        st.write("## Collaborative Filtering recommendation")
    
    case "Hybrid recommendation":
        from model.HybridRecommendationSystem import HybridRecommender
        hrs = HybridRecommender()
        hrs.load_database()
        st.write("## Hybrid recommendation")
        user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
        level = st.slider("number of recommendations", 1, 20)
        t = st.slider("Threshold", 5, 20)
        if(st.button('Submit')):
            df_rec_hrs = hrs.hybrid_recommendation(userId=user, threshold= t, number_of_recommendations= level)
            st.dataframe(df_rec_hrs)
    case "Conclusion":
        st.write("## Conclusion,  critical view + perspectives (what could have been done if we had 3 more months)")