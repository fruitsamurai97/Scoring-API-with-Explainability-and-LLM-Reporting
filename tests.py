import pytest
from unittest.mock import patch
from functions import load_data, load_model, load_explainer
from fct_process import extraction, extract_bounds

# Assuming `functions.py` and `fct_process.py` might interact with external services,
# you'd typically mock these interactions in unit tests to not rely on external dependencies.

@pytest.fixture(scope="module")
def loaded_data():
    with patch('your_module.load_data') as mock_load_data:
        mock_load_data.return_value = # Provide a Pandas DataFrame here as a mock return value
        yield mock_load_data()

def test_data(loaded_data):
    assert not loaded_data.empty

@pytest.fixture(scope="module")
def model():
    with patch('your_module.load_model') as mock_load_model:
        mock_load_model.return_value = # Mock return value (e.g., a model object)
        yield mock_load_model()

def test_model(model):
    assert model is not None  # Adjust based on what `load_model` should return

@pytest.fixture(scope="module")
def explainer():
    with patch('your_module.load_explainer') as mock_load_explainer:
        mock_load_explainer.return_value = # Mock return value
        yield mock_load_explainer()

def test_explainer(explainer):
    assert explainer is not None  # Adjust based on what `load_explainer` should return

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
