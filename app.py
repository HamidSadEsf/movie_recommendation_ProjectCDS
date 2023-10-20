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
    section.main .st-dj{
        color: black !important;
    }
    section.main .st-cj, section.main .st-ch{
        color: white !important;
    }
    section.main .st-bg {
        background: transparent;
    }
    section.main .st-bg::hover {
        color: red;
    }
    h1, h2, h3, h4, p{
        color: white !important;
    }
    p, ol, ul, dl, li{
        font-size: 1.3rem !important;
    }

    </style>

    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('Streamlit_Asset/V01_Dark.jpg')

#path = '/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/'
path = 'Streamlit_Asset/recommenders.png'

st.title('Movie Recommendations')
st.write ('A Project by Sviatlana Viarbitskaya & Hamid S. Esfahlani ')

pages = [
    'Introduction',
    'Objectives',
    'Datasets',
    'Targets and variables',
    'Data limitations',
    'Visualization',
    'Pipeline: collaborative-filtering',
    'Pipeline: content-based',
    'Pipeline: hybrid',
    'Content-based recommendation demo',
    'Collaborative filtering demo',
    'Hybrid recommendation demo',
    'Conclusion']
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
            - 138493 users between January 09, 1995 and March 31, 2015
            - The ratings matrix contains values ranging from 0.5 to 5
            ''')
        
        datasets = ['genome-scores.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
        dataset = st.radio('Choose the dataset to get a snippet', datasets, horizontal=True)
        if dataset == 'genome-scores.csv':
            st.write('Size on disk: 308 MB')
            #st.write('The genome-scores is a structured dataset that stores tag relevance scores for movies. It takes the form of a dense matrix, wherein every movie within the genome is assigned a value corresponding to each tag in the genome.')
            #st.write('It serves as a representation of how well movies manifest specific characteristics denoted by tags.')
            st.dataframe(pd.read_csv('data/external/genome-scores.csv', nrows=10))
        if dataset == 'movies.csv':
            st.write('Size on disk: 1,33 MB')
            st.dataframe(pd.read_csv('data/external/movies.csv', nrows=10))
        if dataset == 'ratings.csv':
            st.write('Size on disk: 508 MB')
            st.dataframe(pd.read_csv('data/external/ratings.csv', nrows=10))
        if dataset == 'tags.csv':
            st.write('Size on disk: 15,8 MB')
            st.dataframe(pd.read_csv('data/external/tags.csv', nrows=10))
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
        graphs=['Distribution of tag relevance across movie tags (CB)',
                'Distribution of rating amount according to genre (CB)',
                'Distribution of rating amount among users (CF)',
                'Average rating vs users rating activity (CF)',
                'Long-tail graphs'
                ]
        graph = st.selectbox('Select a graph', graphs)
        if graph == 'Distribution of tag relevance across movie tags (CB)':
            st.write('The tags exhibit varying relevance scores, with most averaging below 0.2 but having high maximum relevance scores, suggesting their specialization and importance in identifying patterns of similarity for content-based modeling in movie recommendations.')
            chart_data_01 = pd.read_csv('data/external/genome-scores.csv').drop(['movieId'], axis=1).groupby(by='tagId').mean().sort_values(by='relevance',ascending=False, ignore_index=True).reset_index().rename(columns={'index':'Tags','relevance':'Avg relevance score'})
            chart_data_02 = pd.read_csv('data/external/genome-scores.csv').drop(['movieId'], axis=1).groupby(by='tagId').max().sort_values(by='relevance',ascending=False, ignore_index=True).reset_index().rename(columns={'index':'Tags','relevance':'Max relevance score'})
            col1, col2,= st.columns(2)
            with col1:
                st.write("Avg relevance of tags")
                st.line_chart(chart_data_01, y='Avg relevance score', x='Tags')
            with col2:
                st.write("Max relevance of tags")
                st.line_chart(chart_data_02, y='Max relevance score', x='Tags')
               
 
        if graph == 'Distribution of rating amount according to genre (CB)':
            st.write("The graphs reveal that user interaction in various genres isn't solely dependent on movie counts, leading to data preclustering and the inclusion of genres as features in the content-based model to capture genre-specific differences in user engagement.")
        if graph == 'Distribution of rating amount among users (CF)':
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
        if graph == 'Average rating vs users rating activity (CF)':
            st.write('The distribution displays notable rating style biases, particularly among "super users," with extremes in average ratings below 2 and above 4, while casual users exhibit the least bias.')
            df_rating = pd.read_csv('data/external/ratings.csv')
            UserAvgRating = df_rating.groupby('userId').agg({'rating':['mean','count']})
            UserAvgRating.columns = ['Avg rating of user', 'users rating frequency']
            UserAvgRating = UserAvgRating[UserAvgRating['users rating frequency']<2000]
            st.scatter_chart(UserAvgRating, y='Avg rating of user', x='users rating frequency')
        if graph == 'Long-tail graphs':
            st.write("The dataset's small fraction of movies received substantial interaction, leading to high sparsity in both the original and processed data. The presence of a long-tail poses a challenge for collaborative filtering recommendation systems as it can lead to recommending the same popular movies to all users, limiting diversity and personalization.")
            st.image(Image.open('Streamlit_Asset/long-tail.png'), caption = 'Long-tail graphs')
    case "Pipeline: content-based":
        st.write("## Pipeline: Content-based")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preprocessing", "clustering", 'modeling', "Interpretation", "Evaluation"])

        with tab1:
            st.markdown('''
                - Pivot genome scores dataframe to obtain a dense matrix containing a score (0-1) between each movie and each tag
                - Add release year from `movies.csv` and normalize it
                - Add genres from `movies.csv` and encode them as a one-hot numeric array
                ''')
            st.dataframe(pd.read_csv('data/processed/df_ContBaseRec.csv', nrows = 5))

        with tab2:
            st.markdown('''
                - Clustering with KMeans
                - investigate the optimal number of clusters                             
            ''')
            subtab1, subtab2, subtab3 = st.tabs(['elbow method graph', 'silhouette coeffecients', 'Distribution of movies among the clusters'])
            with subtab1:
                st.markdown('''As we can see the elbow graph doesn’t show any significant flattening in the curve.                            
                            ''')
                st.image(Image.open('Streamlit_Asset/Elbow_Method.png'))
            with subtab2:
                st.markdown('we can see a significant rise of the coefficient in the 18th cluster. This number was taken as the number of clusters')
                st.image(Image.open('Streamlit_Asset/silhouette_score.png'))
            with subtab3:
                st.image(Image.open('Streamlit_Asset/distribution_cluster.png'))
        with tab3:
            st.markdown('''
                The algorithms used in the content-based model are Nearest Neighbor (NN) and Cosine Similarity (CS) from scikit-learn.
                
                * The modeling consists of the following steps:
                    1. Find out the id of the given movies
                    2. Calculate the mean of all the given movies (mean point)
                    3. Train the model on the whole data set
                    4. Calculate the distance (NN) and the similarity score (CS) of all movies to the mean point
                    5. Remove the given movies from the returned dataset
                    6. Remove all the movies which are not in the same cluster as the given movie from the returned dataset
                    7. Given that X is an integer, return the top X nearest or most similar movies to the given movie.
                
                
                **Diversification** of the result by “post re-rankin”
                ''')
        with tab4:
            st.markdown('''
                - Efforts to interpret the model using SHAP were hindered by unresolved errors
                - An alternative approach to calculate a feature importance proved computationally prohibitive due to the high number of features (1149) and resource constraints
            ''')           
        with tab5:
            st.markdown('''
                - Coverage : 6%
                - Personalization : Personalization: Calculate overlapping recommendations based on user similarity.
            ''')
            st.image(Image.open('Streamlit_Asset/cb_personalization.png'), caption = 'Personlization CB model')
    case 'Pipeline: collaborative-filtering':
        st.write("## Pipeline: collaborative-filtering")

        tab1, tab2, tab3, tab4 = st.tabs(["Preprocessing", "Model selection", "Model tuning", "Evaluation"])

        with tab1:
            st.markdown('''
                - The user base was reduced to 1000 users (plus 4 additional users for a total of 1004).
                - Movies that received less than 100 ratings were removed, resulting in about 600 movies in the final rating matrix.
                ''')
            st.write('final_ratings.csv')
            st.dataframe(pd.read_csv('data/processed/final_ratings.csv', nrows = 5, usecols = ['userId','movieId','rating'] ))

        with tab2:
            st.markdown('''
                - Surprise scikit library
                - Memory-based k-nearest neighbor (KNN) algorithms (KNNBasic, KNNWithMeans)
                - Model-based matrix factorization SVD algorithms (SVD, SVD++)
                ''')
        with tab3:
            st.markdown('''
                - Minimization of RMSE and MAE
                - GridSearchCV and RandomizedSearchCV
                ''')
            st.markdown('''
                | Models         | Default MAE | Default RMSE  | Tuned MAE | Tuned RMSE |
                | :---           |    :----:   |          ---: |      ---: |       ---: |
                | NormalPredictor| 1.098       | 1.382         | -         | -          |
                | BaselineOnly   | 0.657       | 0.853         | 0.655     |0.851       |
                | KNNBasic       | 0.66        | 0.865         | 0.658     |0.863       |
                | KNNWithMeans   | 0.653       | 0.846         | 0.647     |0.843       |
                | SVD            | 0.637       | 0.831         | 0.617     |0.804       |
                |SVD++           | 0.626       | 0.817         |-          |-           |
            ''')
        with tab4:
            st.markdown('''
                - Accuracy of ratings predictions
                - Coverage: The proportion of items or content in a recommendation system that is actually being recommended to users.
            ''')
            st.image(Image.open('Streamlit_Asset/cf_coverages.png'), caption = 'Coverage of CF models')
            st.markdown('''
                - Personalization: Calculate overlapping recommendations based on user similarity.
                ''')
            st.image(Image.open('Streamlit_Asset/cf_personalization.png'), caption = 'Personlization CF models')
    case 'Pipeline: hybrid':
        st.write("## Pipeline: Hybrid")
        tab1, tab2, tab3 = st.tabs(["Cold-start", "Models", "Evaluation"])
        with tab1:
            st.markdown('''
                - Problem: Struggling to provide recommendations for new users or items with limited data
            ''')
            # latext = r'''
            # ## Latex example
            # ### full equation 
            # $$ 
            # \Delta G = \Delta\sigma \frac{a}{b} 
            # $$ 
            # ### inline
            # Assume $\frac{a}{b}=1$ and $\sigma=0$...  
            # '''
            # st.write(latext)
        with tab2:
            st.markdown('''
                - Ensemble design: Combinatation of CF and CB models
                - Weighted: Computing a weight for the scores from individual ensemble components
                - Switching: Switches between various recommender systems depending on current user
                    - Case 1: If the amount of user’s ratings is lower than a threshold, then we calculate the recommendation from CB and CS  
                    
                    - Case 2: If the user rated more movies than indicated by the threshold, we calculate the recommendation  from CB and CF
            ''')
            
        with tab3:
            st.markdown('''
                - Coverage : 4% (out of more than 10000 movies)
            ''')
            st.image(Image.open('Streamlit_Asset/coverage.png'), caption = 'Coverages')
            st.markdown('''
                - Personalization: Comparison of different models
                ''')
            st.image(Image.open('Streamlit_Asset/personalization.png'), caption = 'Personlizations')
    case "Content-based recommendation demo":
        st.write("## Content Based recommendation demo")
        from model.ContentBasedRec import ContentBasedRecommender
        cbs = ContentBasedRecommender()
        cbs.load_database()
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
        
    case "Collaborative filtering demo":
        st.write("## Collaborative Filtering Demo")

        from model.CollaborativeFilteringRec import CollaborativeFilteringRecommender
        cfr = CollaborativeFilteringRecommender()
        user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
        level = st.slider("number of recommendations", 1, 20)
        if(st.button('Submit')):
            df_rec_cfr = cfr.recommend(user, level, level).drop(['userId'], axis = 1)
            df_rec_cfr = df_rec_cfr.merge(pd.read_csv('data/external/movies.csv'), on='movieId').drop(['movieId'], axis = 1)
            df_rec_cfr.index = pd.RangeIndex(1, level+1)
            st.dataframe(df_rec_cfr)
    case "Hybrid recommendation demo":
        st.write("## Hybrid recommendation demo")
        from model.HybridRecommendationSystem import HybridRecommender
        hrs = HybridRecommender()
        hrs.load_datasets()
        user = st.selectbox('UserId', (pd.read_csv('data/processed/final_ratings.csv').userId) )
        level = st.slider("number of recommendations", 1, 20)
        t = st.slider("Threshold", 5, 20)
        if(st.button('Submit')):
            df_rec_hrs = hrs.hybrid_recommendation(userId=user, threshold= t, number_of_recommendations= level).drop(['movieId'], axis = 1)
            df_rec_hrs.index = pd.RangeIndex(1, level+1)
            st.dataframe(df_rec_hrs)

    case "Conclusion":
        st.write("## Conclusion,  critical view + perspectives (what could have been done if we had 3 more months)")