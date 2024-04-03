
import streamlit as st
from functions import load_data, load_model, load_explainer
from fct_process import extraction, extract_bounds
import pytest
import os
from dotenv import load_dotenv
load_dotenv() 
account_key=os.getenv('AZURE_TEST_ACCOUNT_KEY')


def test_data():
    assert not load_data(account_key).empty

df=load_data(account_key)
def test_model():
    assert load_model()
def test_explainer():
    assert load_explainer()
def test_client_overview():
    cols=["SK_ID_CURR","IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA","CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH"]
    assert set(cols).issubset(set(df.columns))

def test_proba():
    list_IDS = df["SK_ID_CURR"].unique().tolist()
    condition = df['SK_ID_CURR'] == list_IDS[0]
    elt = df[condition]["PAYBACK_PROBA"].tolist()[0] 
    assert elt >= 0 
    assert elt<=1 

def test_extract_bonds():
    assert all(item is not None and item!=""  for item in  extract_bounds(" 0< TEST_LOL <= 1 "))




