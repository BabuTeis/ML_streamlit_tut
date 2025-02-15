import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('🐧 This is a machine learning app, made by BabuTeis.')

st.markdown('This website is an **interactive machine learning app designed '
            'to classify penguin species** based on user-provided features '
            'such as island, bill length, bill depth, flipper length, body '
            'mass, and gender.')

st.markdown('Users can input these features using intuitive '
            'sliders and dropdowns, and the app predicts the penguin species '
            'using a trained Random Forest model. It also displays the '
            'prediction probabilities for each species and allows users '
            'to explore and visualize the underlying dataset, making it '
            'an engaging tool for understanding machine learning in action.')

# Information about the data
with st.expander('Data Information'):
    st.write('**Orginal Dataset:**')
    st.link_button('Kaggle penguin dataset',
                   'https://www.kaggle.com/code/parulpandey/'
                   'penguin-dataset-the-new-iris')

    st.write('**Cleaned Dataset:**')
    st.link_button('Cleaned dataset by DataProfessor',
                   'https://raw.githubusercontent.com/'
                   'dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

    st.write('**Penguin Species**')
    st.image('https://education.rstudio.com/blog/2020/'
             '07/palmerpenguins-cran/gorman-penguins.jpg')

    st.write('**Features Used Information**')
    st.markdown("""
    - **`island`**
    The island where the penguin was observed.

    - **`bill_length_mm`**
    The length of the penguin's bill (beak) in millimeters.

    - **`bill_depth_mm`**
    The depth (thickness) of the penguin's bill in millimeters.

    - **`flipper_length_mm`**
    The length of the penguin's flipper in millimeters.

    - **`body_mass_g`**
    The body mass of the penguin in grams.

    - **`sex`**
    The gender of the penguin.
    """)

with st.expander('Data'):
    st.write('**Raw data**')

    # Load and display the dataset
    df: pd.DataFrame = pd.read_csv('dataset_folder/penguins_cleaned.csv')
    df  # Prints entire dataframe.

    st.write('**X**')
    X_raw: pd.DataFrame = df.drop('species', axis=1)
    X_raw

    st.write('**y**')
    y_raw: pd.Series = df.species
    y_raw

with st.expander('Data visualization'):
    st.write('**Scatterplot Chart**')
    # Future implementation: let users choose variables.
    st.scatter_chart(data=df,
                     x='bill_length_mm',
                     y='body_mass_g',
                     color='species')

# Input features
with st.sidebar:
    st.header('Input features')

    # Get user inputs for the penguin's features
    island: str = st.selectbox('Island', ('Torgersen', 'Dream', 'Biscoe'))
    bill_length_mm: float = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm: float = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm: float = st.slider('Flipper length (mm)',
                                         172.0, 231.0, 201.0)
    body_mass_g: float = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender: str = st.selectbox('Gender', ('male', 'female'))

    # Create a DataFrame for the input features
    data: dict = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df: pd.DataFrame = pd.DataFrame(data, index=[0])
    input_penguins: pd.DataFrame = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    input_df
    st.write('**Combined penguins data**')
    input_penguins

# Data preparation
# Encode categorical columns
encode: list = ['island', 'sex']
df_penguins: pd.DataFrame = pd.get_dummies(input_penguins, prefix=encode)

# Separate the input row and the rest of the dataset
X: pd.DataFrame = df_penguins[1:]
input_row: pd.DataFrame = df_penguins[:1]

# Map target labels to integers
target_mapper: dict = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val: str) -> int:
    """
    Encodes a string target label into an integer using the target_mapper.

    Args:
        val (str): The species name.

    Returns:
        int: Encoded integer for the species.
    """
    return target_mapper[val]


y: pd.Series = y_raw.apply(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    input_row
    st.write('Encoded y')
    y

# Model training and inference
# Train the machine learning model
clf: RandomForestClassifier = RandomForestClassifier()
clf.fit(X, y)

# Make predictions with the trained model
prediction: np.ndarray = clf.predict(input_row)
prediction_proba: np.ndarray = clf.predict_proba(input_row)

# Create a DataFrame for prediction probabilities
df_prediction_proba: pd.DataFrame = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                    1: 'Chinstrap',
                                    2: 'Gentoo'})

# Display predicted species probabilities
st.subheader('Predicted Species')
st.write('The probabilities displayed show the likelihood of '
         'the penguin belonging to each species '
         '(Adelie, Chinstrap, or Gentoo) based on '
         'the input features. The species with the '
         'highest probability is the predicted classification.')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)

# Get the species names for the prediction
penguins_species: np.ndarray = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))

# Show picture of the penguin
if str(penguins_species[prediction][0]) == 'Chinstrap':
    st.image('https://upload.wikimedia.org/wikipedia/'
             'commons/0/08/South_Shetland-2016-Deception_'
             'Island%E2%80%93Chinstrap_penguin_%28Pygoscelis_'
             'antarctica%29_04.jpg', caption='Chinstrap Penguin Photo',
             use_container_width=True)
elif str(penguins_species[prediction][0]) == 'Adelie':
    st.image('https://upload.wikimedia.org/wikipedia/'
             'commons/0/03/Adelie_Penguin.jpg', caption='Adelie Penguin Photo',
             use_container_width=True)
elif str(penguins_species[prediction][0]) == 'Gentoo':
    st.image('https://www.nzbirdsonline.org.nz/assets/'
             '95597/1691015063-24e758e0c6d04f6349e7c84725004114'
             '.jpg?auto=format&fit=crop&w=1200',
             caption='Gentoo Penguin Photo',
             use_container_width=True)


st.markdown('**Version**: *1.00*')
st.markdown('**Latest Update**: *1/24/25*')
