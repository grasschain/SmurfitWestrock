import streamlit as st
import math

# Set page title
st.set_page_config(page_title="Batch Job", layout="centered")

# Title
st.markdown("### IMPORT JOBS HERE:")

# File Uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xlsm"])
if uploaded_file:
    st.success("Upload complete!")

st.markdown("---")  # Horizontal line

#  File Download
("# OPTIMAL STARTING QUANTITIES")

st.download_button(label="DOWNLOAD", data="Sample Data", file_name="xx")

st.write("##### Important features")
st.write("flute code, qty bucket, machine group, blank width")

# Dropdown Selection for Quick View
st.markdown("## QUICK VIEW")
job_selection = st.selectbox("Select Job", ["Job 1", "Job 2", "Job 3"], index=0)

st.write(f"### {job_selection}")

st.write()
st.write()
st.write()