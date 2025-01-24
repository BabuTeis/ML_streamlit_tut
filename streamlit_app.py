import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is a machine learning app, made by BabuTeis.')

with st.expander('Data'):
    st.write('**Raw data**')
    
    df = pd.read_csv('dataset_folder/penguins_cleaned.csv')
    df # Prints entire dataframe.
    
    st.write('**X**')
    X = df.drop('species', axis=1)
    X
    
    st.write('**y**')
    y = df.species
    y
