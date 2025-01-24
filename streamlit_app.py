import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is a machine learning app, made by BabuTeis.')

# df = dataframe
df = pd.read_csv('dataset_folder/penguins_cleaned.csv')
df 
