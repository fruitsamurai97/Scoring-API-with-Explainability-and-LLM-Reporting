#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
#import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

##############################

#############import funcitons
#from functions import make_donut
from functions import load_data
from functions import show_proba
#from functions import import_columns
from functions import client_overview
from functions import show_explanations
#from functions import test_affichage



########## Setting up Azure depot###############
account_name = "fruitsamurai97depot"
account_key=''
with open("azure_container_key.txt", "r") as my_key:
    account_key= my_key.read()
container_name= "assets"

#################################################

################ initialisation affichage ##############
if 'afficher' not in st.session_state:
    st.session_state.afficher = True

################################################

#######################
# Page configuration
st.set_page_config(
    page_title="First app scoring",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

#######################



################ Load data~###########
df= load_data()
#columns_description = import_columns()
col_selection = [c for c in df.columns if c not in ['TARGET','IF_0_CREDIT_IS_OKAY', 'PAYBACK_PROBA','LIME_FEATURES_NAMES','LIME_FEATURES_THRESHOLD','LIME_FEATURES_IMPACT']]
test_df= df[col_selection]
list_IDS = df["SK_ID_CURR"].unique().tolist()
###########################################


#######initialisation des états ############
if 'validate_clicked' not in st.session_state:
    st.session_state['validate_clicked'] = False
if 'credit_overview_clicked' not in st.session_state:
    st.session_state["credit_overview_clicked"] = False

if 'show_exp_clicked' not in st.session_state:
    st.session_state["show_exp_clicked"]= False

####################################################
    
# Création du sidebar
with st.sidebar:
    st.title('Client Overview')    
    selected_ID = st.selectbox("Select client ID", list_IDS)
    # Bouton de validation
    validate_button = st.button("Validate Selection")
    if validate_button:
        st.session_state['validate_clicked'] = True
        st.session_state.afficher = not st.session_state.afficher
    ### création de l'état True pour le bouton cliqué
        
####Création de la mise en page ##########
col1, col2 = st.columns((3,4), gap='medium')
###########################################


    

if 'validate_clicked' in st.session_state and st.session_state["validate_clicked"]:
    with col1:
        client_overview(selected_ID)
        scol1,scol2,scol3 = st.columns((1,2,1))
    #affichage du bouton qui permet d'afficher l'état du crédit
        with scol1:
            st.write("")
        with scol2: 
            credit_overview_bouton= st.button("Simulate Credit Overview")
        with scol3:
            st.write("")
        del scol1,scol2, scol3
### Création état True pour le bouton cliqué
        if credit_overview_bouton:
                st.session_state.afficher = not st.session_state.afficher
                st.session_state["credit_overview_clicked"] = True
        if st.session_state.afficher:
######### Affichage des probas et du bouton show explanations
            if "credit_overview_clicked" in st.session_state and st.session_state["credit_overview_clicked"]:
            # afficher les probas
                show_proba(selected_ID)

                # centrer le bouton et l'afficher 
                scol1,scol2 = st.columns((1,2))
                with scol1:
                    st.write("")
                with scol2:
                    show_explanations_button = st.button("Show Explanations")
    ## Création état True pour le bouton cliqué 
                if show_explanations_button:
                    st.session_state["show_exp_clicked"] = True

                    if 'show_exp_clicked' in st.session_state:
                        #st.session_state["credit_overview_clicked"] = True
                        with col2:
                            #test_affichage(df)
                            show_explanations(selected_ID)


  #for key in st.session_state.keys():
   # del st.session_state[key]    
#alt.themes.enable("dark")
#######################
# Sidebar
# Initialisation de la variable d'état si elle n'existe pas
