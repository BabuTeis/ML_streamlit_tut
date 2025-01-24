import streamlit as st
import pandas as pd
import numpy as np

st.title('🤖 Machine Learning App')

st.info('This is a machine learning app, made by BabuTeis.')

with st.expander('Data'):
    st.write('**Raw data**')

    df = pd.read_csv('dataset_folder/penguins_cleaned.csv')
    df  # Prints entire dataframe.

    st.write('**X**')
    X = df.drop('species', axis=1)
    X

    st.write('**y**')
    y = df.species
    y

with st.expander('Data visualization'):
    st.write('**Scatterplot Chart**')
    # Future implementation: let use choose variables.
    st.scatter_chart(data=df,
                     x='bill_length_mm',
                     y='body_mass_g',
                     color='species')

# Data preparations
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island',
                          ('Torgersen', 'Dream', 'Biscoe'))
    bill_length_mm = st.slider('Bill length (mm)',
                               32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)',
                              13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)',
                                  172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)',
                            2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender',
                          ('male', 'female'))

    # Create a DataFrame for the input features:
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    input_df
    st.write('**Combined penguins data**')
    input_penguins

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]
