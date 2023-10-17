import streamlit as st
import pandas as pd
import numpy as np
import base64
from PIL import Image

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    section.main {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    color: white;
    }

    section.main div{
        color: white !important;
    }

    h1, h2, h3, h4, p
        color: white !important;
    }
    p, ol, ul, dl, li, li::marker {
        font-size: 1.3rem !important;
    }
    </style>

    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('Streamlit_Asset/V01_Dark.jpg')


st.title('Movie Recommendations')
st.write ('A Project by Sviatlana Viarbitskaya & Hamid S. Esfahlani ')

pages = ['Introduction', 'Objectives', 'Datasets', 'Targets and variables', 'Data limitations','Preprocessing', 'Visualization',  'Content based recommendation', 'Collaborative Filtering', 'Hybrid recommendation', 'Conclusion']
page = st.sidebar.radio('Go to', pages)

match page:
    case "Introduction":

        st.write("## Introduction")

        #from PIL import Image
        #st.image(Image.open('Streamlit_Asset/V01_light.jpg'), caption = 'Movie recommendation system')

        #st.markdown('''---''')
        st.markdown("### A  recommendation system:")
        st.markdown('''
            - provides personalized recommendations
            - predicts user preferences
            - helps user to discover products, service, content etc.
            ''')
        st.markdown("### Types of recommenders:")
        st.image(Image.open('Streamlit_Asset/recommenders.png'), caption = 'Type of Recommenders')

        # st.markdown(
        #     """
        #     <style>
        #     section.main {
        #         background: url("Streamlit_Asset/V01_light.jpg")
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )
    case "Objectives":
        st.write("## Objectives")
        st.markdown('''
            - Understand and implement the two concepts of content-based (CB) and user-based collaborative filtering (CF) in two different engines.
            - Develop a hybrid movie recommendation system that combines these two engines to provide more diverse and still accurate recommendations.
            - Evaluate the effectiveness of the hybrid recommendation system through quantitative metrics such as accuracy, diversity and personalisation.
            ''')

    case "Datasets":
        st.write("## Datasets")
        st.write('''
            - 20M dataset of the MovieLens database
            - 20000263 ratings and 465564 tag applications across 27278 movies
            - 138493 users between January 09, 1995 and March 31, 2015.
            ''')
        
        datasets = ['genome-scores.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
        dataset = st.radio('Choose the dataset to get a snippet', datasets, horizontal=True)
        if dataset == 'genome-scores.csv':
            st.write('Size on disk: 308 MB')
            #st.write('The genome-scores is a structured dataset that stores tag relevance scores for movies. It takes the form of a dense matrix, wherein every movie within the genome is assigned a value corresponding to each tag in the genome.')
            #st.write('It serves as a representation of how well movies manifest specific characteristics denoted by tags.')
            st.dataframe(pd.read_csv('data/external/genome-scores.csv').head(10))
        if dataset == 'movies.csv':
            st.write('Size on disk: 1,33 MB')
            st.dataframe(pd.read_csv('data/external/movies.csv').head(10))
        if dataset == 'ratings.csv':
            st.write('Size on disk: 508 MB')
            st.dataframe(pd.read_csv('data/external/ratings.csv').head(10))
        if dataset == 'tags.csv':
            st.write('Size on disk: 15,8 MB')
            st.dataframe(pd.read_csv('data/external/tags.csv').head(10))
    case "Targets and variables":
        st.write("## Targets and Variables")
        st.write("### Content-based")
        st.write("**Features:** tags, release year, movie genres")
        st.write("**Target:** none")
        st.write("### Collaborative filtering")
        st.write("No clear distinction does not exist between target variables and feature variables because each feature plays the dual role of a dependent and independent variable.")

    case "Data limitations":
        st.write("## Data limitations")
        st.markdown('''
            - Very uneven user interactions throughout the period of data acquition
            - A large portion of movies is not properly tagged
            - High sparsity of the rating matrix
            ''')

    case 'Visualization':
        st.write("## Data visualization")
        st.write('Below we present some visualizations that informed the modeling part of the project. ')
        graphs=['Distribution of tag relevance across movie tags',
                'Distribution of rating amount according to genre (CB)',
                'Distribution of rating amount among users (CF)',
                'Avg rating vs users rating activity',
                'Long-tail graphs'
                ]
        graph = st.selectbox('visualization', graphs, placeholder='Select a graph')
        if graph == 'Distribution of tag relevance across movie tags':
            st.write('We can see that the average relevance of more than 80% of tags is below 0.2, which is relatively low. But nearly all tags have a max relevance score of over 0.8. This indicates that most of the tags are specialized. Thus a data reduction upon missing relevancy of the tags is not possible. But also we can deduce that the tags offer a good reference to find patterns of similarity between movies which will be the base of the CB modeling.')
            chart_data_01 = pd.read_csv('data/external/genome-scores.csv').drop(['movieId'], axis=1).groupby(by='tagId').mean().sort_values(by='relevance',ascending=False, ignore_index=True).reset_index().rename(columns={'index':'Tags','relevance':'Avg relevance score'})
            chart_data_02 = pd.read_csv('data/external/genome-scores.csv').drop(['movieId'], axis=1).groupby(by='tagId').max().sort_values(by='relevance',ascending=False, ignore_index=True).reset_index().rename(columns={'index':'Tags','relevance':'Max relevance score'})
            col1, col2,= st.columns(2)
            with col1:
                st.write("Avg relevance of tags")
                st.line_chart(chart_data_01, y='Avg relevance score', x='Tags')
            with col2:
                st.write("Max relevance of tags")
                st.line_chart(chart_data_02, y='Max relevance score', x='Tags')
               
 
        if graph == 'Distribution of rating amount according to genre':
            st.write('The juxtaposition of these graphs shows that the user interaction within certain genres is higher than others and doesn’t directly relate to the Movie count of that genre. This indicates certain niches of high user interaction, i.a. Sci-Fi. While genres with great movie count contain also a lot of movies with low interaction. This conclusion led us to precluster our data before training it on a ML model.Also, the genre specific difference in interaction led to the integration of genres as features for the CB Model.')
        if graph == 'Distribution of rating amount among users':
            st.write(' The graph below shows the strong presence of outliers in our data - users that rated 10 to 100 times more movies than the average user of the MovieLens platform - the so called super users.')
            import matplotlib.pyplot as plt
            import seaborn as sns
            df = pd.read_csv('data/external/ratings.csv').userId.value_counts()
            plt.figure(figsize=(20, 1))
            p= sns.boxplot(x=df)
            p.set(title='Distribution of rating amount of users', xlabel='amount of ratings', ylabel='users', xlim=[-50,9500])
            fig = p.get_figure()
            st.pyplot(fig)
            st.write('The graph below is just a zoom of the above graph, from which we can see that the majority of the users rated between 40 and 150 movies, with the median around 70.')
            p.set(xlim=(0,350), xticks=range(0,350,10))
            fig = p.get_figure()
            st.pyplot(fig)
        if graph == 'Avg rating vs users rating activity':
            st.write('The distribution exhibits a significant bias in the ratings styles of different users, extremes being users whose average rating is below 2 and above 4. Interestingly, it is the super users that tend to show the largest bias, while a casual user mean rating tends to show the least bias. The effect of this bias will be discussed in the corresponding modeling section.')
            df_rating = pd.read_csv('data/external/ratings.csv')
            UserAvgRating = df_rating.groupby('userId').agg({'rating':['mean','count']})
            UserAvgRating.columns = ['Avg rating of user', 'users rating frequency']
            UserAvgRating = UserAvgRating[UserAvgRating['users rating frequency']<2000]
            st.scatter_chart(UserAvgRating, y='Avg rating of user', x='users rating frequency')
        if graph == 'Long-tail graphs':
            st.write('The long-tail graphs show the distribution of ratings among movies in the original and preprocessed ratings dataset. Movies on the left side are called as popular because their popularity is higher than those in long-tail areas. We can see the original data contains a large number of unpopular movies and only a fraction of movies receive a significant number of ratings from users. The situation is very different for the case of the typical reduced ratings dataset. While we can still identify popular and unpopular items, the difference between them is less striking.')
            st.write('The small fraction of movies present in the data set has experienced a significant interaction. This fact is reflected in the high sparsity of the original and processed data. The presence of a long-tail is a challenge for a collaborative filtering recommendation system that wants to be diverse and personalized: the strong presence of popular items will result in that the system will recommend the same popular movies to all the users.')
            
    case "Preprocessing":
        st.write("## Preprocessing")
        model = st.radio('Choose the model', ['Content-based', 'Collaborative filtering'], horizontal=True)
        if model == 'Content-based':
            st.markdown('''
                - Pivot genome scores dataframe to obtain a dense matrix containing a score (0-1) between each movie and each tag
                - Add release year from movies dataset and normalize it
                - Add 20 genres
                ''')
            st.dataframe(pd.read_csv('data/processed/df_ContBaseRec.csv').head(5))
        if model == 'Collaborative filtering':  
            st.markdown('''
                - The user base was reduced to 1000 users (plus 4 additional users for a total of 1004).
                - Movies that received less than 100 ratings were removed, resulting in about 600 movies in the final rating matrix.
                - The ratings matrix contains values ranging from 0.5 to 5.
                ''')
            st.write('final_ratings.csv')
            st.dataframe(pd.read_csv('data/processed/final_ratings.csv').tail(5))
            
    case "Content based recommendation":
        from model.ContentBasedRec import ContentBasedRecommender
        cbs = ContentBasedRecommender()
        cbs.load_database()
        st.write("## Content Based recommendation")
        status = st.radio("Based on: ", ('UserId', 'Movies'))
        if(status== "UserId"):
            user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
            level = st.slider("number of recommendations", 1, 20)
            if(st.button('Submit')):
                df_rec_cbs = cbs.recommendation(userId= user, number_of_recommendations= level).drop(['movieId','labels'], axis = 1)
                df_rec_cbs.index = pd.RangeIndex(1, level+1)
                st.dataframe(df_rec_cbs)
        else:
            movies = st.multiselect('Enter your movies', (pd.read_csv('data/processed/df_labeledMovies.csv').title) )
            level = st.slider("number of recommendations", 1, 20)
            if(st.button('Submit')):
                df_rec_cbs = cbs.recommendation_movies(given_movies= movies, number_of_recommendations= level).drop(['movieId','labels'], axis = 1)
                df_rec_cbs.index = pd.RangeIndex(1, level+1)
                st.dataframe(df_rec_cbs)
        
    case "Collaborative Filtering":
        from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender
        cfr = CollaborativeFilteringRecommender()
        cfr.fit_and_predict()
        st.write("## Collaborative Filtering recommendation")
        user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
        level = st.slider("number of recommendations", 1, 20)
        if(st.button('Submit')):
            df_rec_cfr = cfr.recommend(user, level).drop(['userId'], axis = 1)
            df_rec_cfr = df_rec_cfr.merge(pd.read_csv('data/external/movies.csv'), on='movieId').drop(['movieId'], axis = 1)
            df_rec_cfr.index = pd.RangeIndex(1, level+1)
            st.dataframe(df_rec_cfr)
    case "Hybrid recommendation":
        from model.HybridRecommendationSystem import HybridRecommender
        hrs = HybridRecommender()
        hrs.load_datasets()
        st.write("## Hybrid recommendation")
        user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
        level = st.slider("number of recommendations", 1, 20)
        t = st.slider("Threshold", 5, 20)
        if(st.button('Submit')):
            df_rec_hrs = hrs.hybrid_recommendation(userId=user, threshold= t, number_of_recommendations= level).drop(['movieId'], axis = 1)
            df_rec_hrs.index = pd.RangeIndex(1, level+1)
            st.dataframe(df_rec_hrs)
    case "Conclusion":
        st.write("## Conclusion,  critical view + perspectives (what could have been done if we had 3 more months)")