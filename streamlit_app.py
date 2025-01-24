import streamlit as st
import pandas as pd
import numpy as np

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

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
