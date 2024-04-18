import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from functions import features_client


list_features=["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", 
                "AMT_GOODS_PRICE", "DAYS_BIRTH"]


if 'list_IDS' in st.session_state:
    st.write("IDs available in features.py:", st.session_state['last_selected_ID'])
    selected_ID = st.sidebar.selectbox("Select client ID", st.session_state['list_IDS']
                    , index=st.session_state['list_IDS'].index(st.session_state['last_selected_ID']))
    selected_feature = st.selectbox("Select client ID",list_features)
    features_client(selected_ID,selected_feature)
    if selected_ID != st.session_state["last_selected_ID"]:
        st.session_state['last_selected_ID'] = selected_ID
else:
    st.error("Client IDs not loaded properly.")
