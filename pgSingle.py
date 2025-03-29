import streamlit as st
import math

# Set page title
st.set_page_config(page_title="Single Job", layout="centered")

# Title
st.markdown("Enter Job Information:")

left, middle, right = st.columns(3, vertical_alignment="top")

left._number_input("Order Qty", min_value=0)
with middle:
    job1 = st.selectbox("Select Machine 1", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job2 = st.selectbox("Select Machine 2", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job3 = st.selectbox("Select Machine 3", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job4 = st.selectbox("Select Machine 4", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
    job5 = st.selectbox("Select Machine 5", ["Asitrade", "Digital", "Diecutter", "Flexopress", "Gluer", "Gopfert"],
                        index=None,
                        placeholder="Select machine group")
with right:
    st.write("Input feature information",)
    st.button(label="Clear All", key=None, help=None, on_click="pgBatch.py",
         type="secondary", icon=None, disabled=False, use_container_width=True)

st.markdown("---")  # Horizontal line

#  File Download
("# OPTIMAL STARTING QUANTITIES")



# Dropdown Selection for Quick View
st.write()
st.write()
st.write()