import streamlit as st
import math

orderQty=0

# Set page title
st.set_page_config(page_title="Batch Job", layout="centered")

if st.button(label="Return to Home Page", key=None, help=None, type="secondary", icon=None,
            disabled=False, use_container_width=False):
    st.switch_page("pgTitle.py")

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
job_selection = st.selectbox("Select Job to View", ["Job 1", "Job 2", "Job 3"], index=None, placeholder="Select a job")

st.write(f"### {job_selection}")

if job_selection == "Job 1":
    orderQty = 10000
elif job_selection == "Job 2":
    orderQty = 20000
elif job_selection == "Job 3":
    orderQty = 30000

wasteQty = round(0.05 * orderQty)
optQty = round(orderQty - wasteQty)
finalQty = round(orderQty + (0.001 * orderQty))

#  Information Return
# Starting Quantity
st.write("### Start with: ", optQty)

# Estimated amount of waste
st.write("Estimated Waste: ", wasteQty)

# Finished Quantity
st.write("Final Output Quantity: ", finalQty)