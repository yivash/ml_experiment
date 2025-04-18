from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

#Visual
import streamlit as st

st.set_page_config(
    page_title="ML experiment",
    page_icon=":bar_chart:",
    layout="wide")

# Load the model
model_path = Path(__file__).parent.parent / "models" 
models = os.listdir(model_path)

st.title(f"ML experiment: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"Models available: {models}")

model = joblib.load(model_path / models[0])
st.write(f"Model loaded: {model}")
