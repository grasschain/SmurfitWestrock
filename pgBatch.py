import streamlit as st
import math
import os

orderQty=0

# FIRST SECTION

# Set page title
st.set_page_config(page_title="Batch Job", layout="centered")

if st.button(label="Return to Home Page", key=None, help=None, type="secondary", icon=None,
            disabled=False, use_container_width=False):
    st.switch_page("pgTitle.py")

# Title
st.markdown("### IMPORT JOBS HERE:")

# File Uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    st.success("Upload complete!")

if uploaded_file != None:
    os.rename(uploaded_file.name, "JobsToPredict.xlxs")
    with open("phase1.py") as f:
        exec(f.read())
    with open("phase2.py") as g:
        exec(g.read())

# SECOND SECTION
st.markdown("---")  # Horizontal line

#  File Download
("# OPTIMAL STARTING QUANTITIES")

st.download_button(label="DOWNLOAD", data="Job_Machine_Quantities.xlsx", file_name="Job_Machine_Quantities.xlsx")

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