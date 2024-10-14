import streamlit as st
from streamlt import StreamLit_App

# Streamlit app with two tabs
st.title("🚗 Business Parking and Image Labeling with Interpretability")
# Create tabs
tab1, tab2 = st.tabs(["CSV Prediction (Parking)", "Image Prediction (Label)"])
StreamLit_App().tabOne(tab1)
StreamLit_App().tabTwo(tab2)
