import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_parquet():
    test_set_path = BASE_DIR/"app_info/test_set.parquet"
    test_df = pd.read_parquet(test_set_path)
    test_df = test_df.reset_index(drop=True)
    
    return test_df
