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
from functions import client_overview
from functions import load_explanations
from functions import show_proba, show_explanations, highlight_instance
from functions import fetch_ids, create_prompt

#######################
# Page configuration
st.set_page_config(
    page_title="First app scoring",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
    )




#######Initialize states ############

if 'display_flag' not in st.session_state:
    st.session_state["display_flag"] = True

if 'list_IDS' not in st.session_state:
    st.session_state['list_IDS'] = fetch_ids()

if 'last_selected_ID' not in st.session_state:
    st.session_state['last_selected_ID'] = st.session_state['list_IDS'][0]
    #st.session_state['explanations'] = load_explanations(st.session_state['last_selected_ID'])


if 'show_exp_clicked' not in st.session_state:
    st.session_state["show_exp_clicked"]= False


####################################################
   
#Sidebar creation
with st.sidebar:
    st.title('Options')  
    #selected_ID = st.sidebar.selectbox("Select client ID", st.session_state['list_IDS'])
    selected_ID = st.sidebar.selectbox("Select client ID", st.session_state['list_IDS']
                , index=st.session_state['list_IDS'].index(st.session_state['last_selected_ID']))

    # Vérification si l'ID sélectionné a été re-sélectionné
    if selected_ID != st.session_state['last_selected_ID']:
        st.session_state["display_flag"] =False
        st.session_state['last_selected_ID'] = selected_ID
     


        
####Set up page form ##########
col1, col2 = st.columns((3,4), gap='medium')
###########################################
    
with col1:
    client_overview(selected_ID)
with col2: 
    st.markdown("#### Probabilities estimation")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    show_proba(selected_ID)
    scol1,scol2,scol3=st.columns((1,3,1))
    with scol1:
        st.markdown("")
        st.markdown("")
        show_explanations_button = st.button("Show Explanations")
    del scol1,scol2,scol3
## Création état True pour le bouton cliqué 
    if show_explanations_button:
        st.session_state["display_flag"] = True   
        st.session_state["show_exp_clicked"] = True
    
    if "display_flag" in st.session_state and st.session_state["display_flag"]:
        if 'show_exp_clicked' in st.session_state and st.session_state["show_exp_clicked"]:
            
            features_names,lime_threshold, features_impact, exp_list = load_explanations(selected_ID)
            with col1:
                st.markdown("#### Display a feature")  
                selected_feature= st.selectbox('Choose a feature to display', features_names) 
                highlight_instance(selected_ID,selected_feature)
        
            with col2:
                st.markdown("")
                st.markdown("")
                show_explanations(selected_ID)
            create_prompt(selected_ID)
