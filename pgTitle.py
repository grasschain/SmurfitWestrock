import streamlit as st
import math

#Startup Page
st.image("swLogo.png", caption=None, width=200)
st.title("QUANTITY OPTIMIZER")

st.button(label="Single", key=None, help=None, on_click="pgSingle.py",
         type="primary", icon=None, disabled=False, use_container_width=True)
st.button(label="Batch", key=None, help=None, on_click="pgBatch.py",
         type="primary", icon=None, disabled=False, use_container_width=True)