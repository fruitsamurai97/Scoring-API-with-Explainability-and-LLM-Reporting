#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
#import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
##############################

#############import funcitons
#from functions import make_donut
from functions import load_data
from functions import show_proba
from functions import import_columns
from functions import client_overview
from functions import show_explanations

#######################
# Page configuration
st.set_page_config(
    page_title="First app scoring",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

#######################



################ Load data~###########
df, train_df= load_data()
columns_description = import_columns()
col_selection = [c for c in df.columns if c not in ['TARGET', 'credit accordé == 0', 'Proba de remboursement']]
test_df= df[col_selection]
list_IDS = df["SK_ID_CURR"].unique().tolist()
###########################################


#alt.themes.enable("dark")
#######################
# Sidebar
# Initialisation de la variable d'état si elle n'existe pas
if 'validate_clicked' not in st.session_state:
    st.session_state['validate_clicked'] = False

# Création du sidebar
with st.sidebar:
    st.title('Client Overview')    
    selected_ID = st.selectbox("Select client ID", list_IDS)
    # Bouton de validation
    validate_button = st.button("Validate Selection")
    if validate_button:
        st.session_state['validate_clicked'] = True
col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')
# Logique conditionnelle basée sur le bouton de validation
if 'validate_clicked' in st.session_state and st.session_state['validate_clicked']:
    with col1:
        show_proba(df, list_IDS, selected_ID) 
    # Création conditionnelle du bouton d'explications basé sur l'état de session
        show_explanations_button = st.button("Voir les explications")
    if show_explanations_button:
        with col2:
            show_explanations(selected_ID)