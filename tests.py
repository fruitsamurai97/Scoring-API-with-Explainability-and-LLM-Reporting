import pytest
from functions import load_data, load_model, load_explainer
from fct_process import extraction, extract_bounds

# Fixture to load data once and use it in multiple tests
@pytest.fixture(scope="module")
def loaded_data():
    return load_data()

def test_data(loaded_data):
    assert not loaded_data.empty

def test_model():
    assert load_model()

def test_explainer():
    assert load_explainer()

def test_client_overview(loaded_data):
    cols = [
        "SK_ID_CURR",
        "IF_0_CREDIT_IS_OKAY",
        "PAYBACK_PROBA",
        "CODE_GENDER",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
    ]
    assert set(cols).issubset(set(loaded_data.columns))

def test_proba(loaded_data):
    list_IDS = loaded_data["SK_ID_CURR"].unique().tolist()
    condition = loaded_data['SK_ID_CURR'] == list_IDS[0]
    elt = loaded_data[condition]["PAYBACK_PROBA"].tolist()[0] 
    assert 0 <= elt <= 1

def test_extract_bounds():
    assert all(item is not None and item != "" for item in extract_bounds(" 0< TEST_extract_bounds <= 1 "))
