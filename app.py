import streamlit as st
import pandas as pd
import numpy as np

st.title('Movie Recommendation System App')

pages = ['Introduction', 'Databases', 'Preprocessing', 'Content based', 'User based', 'Hybrid', 'Conclusion']
page = st.sidebar.radio('Go to', pages)


match page:
    case "Introduction":
        st.write("## Introduction")

    case "Databases":
        st.write("## Data")

    case "Preprocessing":
        st.write("## Preprocessing")

    case "Content based":
        st.write("## Content based recommendation")

    case "User based":
        st.write("## User based recommendation")
    
    case "Hybrid":
        st.write("## Hybrid recommendation")
    
    case "Conclusion":
        st.write("## Conclusion")
