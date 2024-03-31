
import streamlit as st
from functions import load_data, load_model, load_explainer
from fct_process import extraction, extract_bounds
import pytest

################ Load data test ~###########
def test_data():
    assert not load_data().empty

df=load_data()
def test_model():
    assert load_model()
clf = load_model()
def test_explainer():
    assert load_explainer()
explainer= load_explainer()
def test_client_overview():
    cols=["SK_ID_CURR","IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA","CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH"]
    assert set(cols).issubset(set(df.columns))

def test_proba():
    list_IDS = df["SK_ID_CURR"].unique().tolist()
    condition = df['SK_ID_CURR'] == list_IDS[0]
    elt = df[condition]["PAYBACK_PROBA"].tolist()[0] 
    assert elt >= 0 
    assert elt<=1 

def test_explanations(): #test load, show explanations and extractions
    feats = [f for f in df.columns if f not in ['SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index',"IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA"]]
    test_x = df[feats]
    test_x_np = test_x.to_numpy()
    list_IDS = df["SK_ID_CURR"].unique().tolist()
    condition = df['SK_ID_CURR'] == list_IDS[0]
    client_instance = test_x_np[df[condition].index[0]]
      
    exp= explainer.explain_instance(
        data_row=client_instance, 
        predict_fn=clf.predict_proba, 
        num_features=5
    )
    assert all(item is not None and item!=""  for item in extraction(exp.as_list()))

def test_extract_bonds():
    assert all(item is not None and item!=""  for item in  extract_bounds(" 0< TEST_LOL <= 1 "))




