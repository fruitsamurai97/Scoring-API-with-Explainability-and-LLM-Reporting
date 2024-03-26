#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
#import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objs as go
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

##############################

#############import funcitons
#from functions import make_donut
from functions import load_data
from functions import load_model
from functions import client_overview

from functions import show_proba, show_explanations, highlight_instance







#######################
# Page configuration
st.set_page_config(
    page_title="First app scoring",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")



################ Load data and model~###########
df= load_data()
clf = load_model()
list_IDS = df["SK_ID_CURR"].unique().tolist()



#######Initialize states ############

if 'display_flag' not in st.session_state:
    st.session_state["display_flag"] = True

if 'last_selected_ID' not in st.session_state:
    st.session_state.last_selected_ID = None


if 'show_exp_clicked' not in st.session_state:
    st.session_state["show_exp_clicked"]= False

####################################################
    
#Sidebar creation
with st.sidebar:
    st.title('Client Overview')    
    selected_ID = st.selectbox("Select client ID", list_IDS)
    # Vérification si l'ID sélectionné a été re-sélectionné
    if selected_ID == st.session_state.last_selected_ID:
        # Désactiver le flag d'affichage
        not st.session_state["display_flag"] 
    else:
        # Affichage lorsque on change d'ID
        st.session_state["display_flag"] =False
        st.session_state.last_selected_ID = selected_ID
    
        
####Set up page form ##########
col1, col2 = st.columns((3,4), gap='medium')
###########################################
    
with col1:
    client_overview(df, selected_ID)
with col2: 
    st.markdown("#### Probabilities estimation")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    show_proba(df,selected_ID)
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
            
                with col2:
                    st.markdown("")
                    st.markdown("")
                    show_explanations(df,selected_ID,clf)
                with col1:
                    st.markdown("#### Display a feature")
                    highlight_instance(df,selected_ID,clf)

                

