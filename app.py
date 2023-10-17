import streamlit as st
import pandas as pd
import numpy as np

st.title('Movie Recommendation System App')

pages = ['Introduction', 'Datasets', 'Visualization', 'Preprocessing', 'Content based recommendation', 'Collaborative Filtering', 'Hybrid recommendation', 'Conclusion']
page = st.sidebar.radio('Go to', pages)

match page:
    case "Introduction":
        st.write("## Introduction")
        from PIL import Image
        st.image(Image.open('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/Streamlit_Asset/V01_light.jpg'), caption = 'Movie recommendation system')
        st.write ('A Project by Sviatlana Viarbitskaya & Hamid S. Esfahlani ')
        st.markdown('''---''')
        st.markdown('''In this report we describe and analyze our group effort to develop a weighted and switching hybrid movie recommendation system by means of combining two distinct approaches to developing a recommender - a content-based and user-based collaborative filtering. The goal is to determine and recommend a given amount of movies for each user in the dataset with the aim to sustain and increase the user’s engagement with the recommendation platform. The latter is hoped to be achieved through building a diverse and personalized recommendation system based on user rating history (collaborative filtering), analysis of the items features (content-based) and handling of the cold-start problem. Finally, we analyze the performance of the proposed system and discuss the desirable future developments.''')
        st.subheader('''Objectives''')
        st.markdown('''
                    - Understand and implement the two concepts of content-based and user-based collaborative filtering in two different engines.
                    - Develop a hybrid movie recommendation system that combines these two engines to provide more diverse and still accurate recommendations.
                    - Evaluate the effectiveness of the hybrid recommendation system through quantitative metrics such as accuracy, diversity and personalisation.
                    ''')
        
    case "Datasets":
        st.write('We utilized 20M dataset of the MovieLens database (https://grouplens.org/datasets/movielens/20m/). This database contains extensive information about movies and user behavior and is freely available for academic and research purposes.')
        st.write('The dataset contains 20000263 ratings and 465564 tag applications across 27278 movies. These data were created by 138493 users between January 09, 1995 and March 31, 2015. This dataset was generated on October 17, 2016.')
        st.write('We used only the following datasets :')
        datasets = ['genome-scores.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
        dataset = st.radio('Choose the dataset to get a snippet', datasets, horizontal=True)
        if dataset == 'genome-scores.csv':
            st.write('Size on disk: 308 MB')
            st.write('The genome-scores is a structured dataset that stores tag relevance scores for movies. It takes the form of a dense matrix, wherein every movie within the genome is assigned a value corresponding to each tag in the genome.')
            st.write('It serves as a representation of how well movies manifest specific characteristics denoted by tags.')
            st.dataframe(pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/genome-scores.csv').head(10))
        if dataset == 'movies.csv':
            st.write('Size on disk: 1,33 MB')
            st.dataframe(pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/movies.csv').head(10))
        if dataset == 'ratings.csv':
            st.write('Size on disk: 508 MB')
            st.dataframe(pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/ratings.csv').head(10))
        if dataset == 'tags.csv':
            st.write('Size on disk: 15,8 MB')
            st.dataframe(pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/tags.csv').head(10))
        st.write('You can find detailed information about the files structures and content of the datasets above in the documentation: https://files.grouplens.org/datasets/movielens/ml-20m-README.html')

    case 'Visualization':
        st.write('Data visualization')
        st.write('Below we present some visualizations that informed the modeling part of the project. ')
        graphs=['Distribution of tag relevance across movie tags',
                'Distribution of rating amount according to genre',
                'Distribution of rating amount among users',
                'Avg rating vs users rating activity',
                'Long-tail graphs'
                ]
        graph = st.selectbox('visualization', graphs, placeholder='Select a graph')
        if graph == 'Distribution of tag relevance across movie tags':
            st.write('We can see that the average relevance of more than 80% of tags is below 0.2, which is relatively low. But nearly all tags have a max relevance score of over 0.8. This indicates that most of the tags are specialized. Thus a data reduction upon missing relevancy of the tags is not possible. But also we can deduce that the tags offer a good reference to find patterns of similarity between movies which will be the base of the CB modeling.')
            chart_data_01 = pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/genome-scores.csv').drop(['movieId'], axis=1).groupby(by='tagId').mean().sort_values(by='relevance',ascending=False, ignore_index=True).reset_index().rename(columns={'index':'Tags','relevance':'Avg relevance score'})
            chart_data_02 = pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/genome-scores.csv').drop(['movieId'], axis=1).groupby(by='tagId').max().sort_values(by='relevance',ascending=False, ignore_index=True).reset_index().rename(columns={'index':'Tags','relevance':'Max relevance score'})
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
            df = pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/ratings.csv').userId.value_counts()
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
            df_rating = pd.read_csv('/home/mumu/Documents/DS/movie_recommendation_ProjectCDS/data/external/ratings.csv')
            UserAvgRating = df_rating.groupby('userId').agg({'rating':['mean','count']})
            UserAvgRating.columns = ['Avg rating of user', 'users rating frequency']
            UserAvgRating = UserAvgRating[UserAvgRating['users rating frequency']<2000]
            st.scatter_chart(UserAvgRating, y='Avg rating of user', x='users rating frequency')
        if graph == 'Long-tail graphs':
            st.write('The long-tail graphs show the distribution of ratings among movies in the original and preprocessed ratings dataset. Movies on the left side are called as popular because their popularity is higher than those in long-tail areas. We can see the original data contains a large number of unpopular movies and only a fraction of movies receive a significant number of ratings from users. The situation is very different for the case of the typical reduced ratings dataset. While we can still identify popular and unpopular items, the difference between them is less striking.')
            st.write('The small fraction of movies present in the data set has experienced a significant interaction. This fact is reflected in the high sparsity of the original and processed data. The presence of a long-tail is a challenge for a collaborative filtering recommendation system that wants to be diverse and personalized: the strong presence of popular items will result in that the system will recommend the same popular movies to all the users.')
            
    case "Preprocessing":
        st.write("## data preparation pipeline")
        st.write('To enrich our movie information, we added release year and genres as categorical values to the dataset.')
        model = st.radio('Choose the model', ['Content based recommendation', 'Collaborative filtering recommendation'], horizontal=True)
        if model == 'Content based recommendation':
            st.write('# Cleaning and processing the data')
            st.write('The genome scores dataset (genome-scores.csv) provides a clear and NaN-less dataset as a matrix. To be able to process the data, we pivot the data frame so each row represents a movie and the columns the tags, as can be seen in the figure below.')
            st.dataframe(pd.read_csv('data/processed/df_ContBaseRec.csv').head(5))
            st.write('The result is a dense matrix containing a score between each movie and each tag. The score is a float between 0-1.')
            st.write('In a further step we added the release year and genres as columns to this data set. These two values had to be extracted from the movies.csv. The release year had to be extracted from the title of each movie. The format of the titles is not always consistent. So some manual problem detection and adjustments had to be made, in order to extract all release years accurately.')
            st.write('The 20 different genres were integrated into the matrix as an indicator variable for each genre. In other words 20 columns were added to the final dataframe, each representing a genre.')
            st.write('# Normalization/standardization of data')
            st.write('Since all the values in the gnome matrix are distributed between 0-1, we had to normalize our additional data, namely genres and release year. Genres were presented as indicator variables of 0 and 1. So there was no need for normalization. Release year is a continuous variable between 1890 and 2015. To be able to integrate these values into our matrix we normalized the release year by applying a min–max normalization.')
            st.write('# Dimension reduction')
            st.write('No dimension reduction was applied, since the provided gnome dataset already effectively reduces the amount of movies. However, in a further process of interpreting the model, a feature reduction seemed to be necessary due to the amount of features and the computational time of the model interpretation models, which was discarded, since the  PCA, which was the preferred dimension reduction method, doesn’t allow us to identify the importance of the original Features.')
        if model == 'Collaborative filtering recommendation':  
            st.write('# Cleaning and processing the data')
            st.write('The ratings data set is originally clean and without duplicates. After having filtered the movies that are not present in the genome-score dataset, we proceeded with further shrinking the user base down to 1000 users. This number approximately corresponds to a maximal number of users for which the collaborative filtering algorithms could have been run locally without crashing the Anaconda application. Additionally, we added 2 users corresponding to Sviatlana and Hamid (the ratings could have been downloaded from the corresponding profiles on the MovieLens website), and 2 users corresponding to fixious users with the tastes opposite that of Sviatlana and Hamid, thus resulting in 1004 user base.')
            st.write('Following these operations, the corresponding movie base has been reduced significantly (movies that have not been ranked by these reduced set users are not taken into account during the modeling stages). Finally, we removed the movies that have been rated less than 100 times, resulting in about 600 movies in the final rating matrix.')
            st.write('Such reduced ratings dataset can be directly used for modeling with the Surprise collaborative filtering library.')
            st.write('# Normalization/standardization of data')
            st.write('The ratings matrix is a sparse matrix and contains the values of ratings ranging from 0.5 till 5. Since we are using the Surprise library, no preliminary normalization or standardization of the data is necessary? However, certain tasted algorithms would perform the data transformation internally, e.g the KNNWithMeans algorithm would normalize each user’s ranking to its mean, thus reducing rating biases in the original data.')
            st.write('# Dimension reduction')
            st.write('The final version of the user-based collaborative filtering indeed uses the matrix-factorization based SVD algorithm, which computes the finite number of latent factors for computation of user similarities. This technique can be considered as dimension reduction technique. The main reason for using SVD in this project is because it is quicker than KNN based algorithms, while its accuracy is comparable to the latter.')
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