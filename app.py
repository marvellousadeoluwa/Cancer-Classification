import streamlit as st
import pickle
import numpy as np
import re
from pathlib import Path
import helper

final_features = helper.final_features
__version__ = helper.__version__


st.write("""
# Breast Cancer Classification
**Welcome to this cancer classification app.**
The objective is to minimize manual involvement in detection of cancerous lumps.
For more information visit: *'https://www.hindawi.com/journals/wcmc/2019/5176705/tab/'*
""")

st.text('Please do well to input the following as required: ')
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f'{BASE_DIR}/breast_cancer_model-{__version__}.pkl', 'rb') as f:
    model = pickle.load(f)

mean_concavity = float(st.number_input('mean concavity'))
mean_concave_points = float(st.number_input('mean concave points'))

worst_radius = float(st.number_input('worst radius'))
worst_perimeter = float(st.number_input('worst_perimeter'))

worst_area = float(st.number_input('worst area'))
worst_concave_points = float(st.number_input('worst concave points'))


detect_btn = st.button('Detect if Breast Cancer')
feats = [mean_concavity, mean_concave_points, worst_radius, worst_perimeter, worst_area, worst_concave_points]

if detect_btn:
    result = helper.predict_cancer(model = model, features= feats)
    if result.lower() == 'benign':
        st.text(f'The tumor is {result}. You are safe, it is not cancerous.\n You just take proper medical care of the lump.')
    elif result.lower() == 'malignant':
        st.text(f'The tumor is {result}. It is cancerous. Treat with urgency.\n Please, proceed to the right medical personnel for treatment.')