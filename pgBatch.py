import streamlit as st
import math
import os
import subprocess
import pandas as pd

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
    saved_path = os.path.join("/mount/src/smurfitwestrock/", uploaded_file.name)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    new_path = "/mount/src/smurfitwestrock/JobsToPredict.xlsx"
    if not os.path.exists(new_path):
        st.error(f"File not found: {new_path}. Please upload the file again.")
    #else:
        #st.success(f"File found: {new_path}")
    os.rename(saved_path, new_path)
    #st.success(f"File renamed to {new_path}")


# SECOND SECTION
st.markdown("---")  # Horizontal line

#  File Download
st.markdown("# OPTIMAL STARTING QUANTITIES")

if uploaded_file:
    if os.path.exists("phase1.py"):
        with open("phase1.py") as f:
            exec(f.read())
    else:
        st.error("phase1.py not found in workspace!")
    if os.path.exists("phase2.py"):
        with open("phase2.py") as f:
            exec(f.read())
    else:
        st.error("phase2.py not found in workspace!")


    st.write("##### Important features")
    st.write('Flute Code Grouped, Qty Bucket, Component Code Grouped, Machine Group 1, Last Operation, qty_ordered, number_up_entry_grouped, OFFSET?, Operation, Test Code')


if uploaded_file:
    download_path = "/workspaces/SmurfitWestrock/Job_Machine_Quantities.xlsx"
    if os.path.exists(download_path):
        with open(download_path, "rb") as f:
            st.download_button(label="DOWNLOAD", data=f, file_name="Job_Machine_Quantities.xlsx")


    # Dropdown Selection for Quick View
    st.markdown("## QUICK VIEW")

    if os.path.exists(download_path):
        df = pd.read_excel(download_path)
    else:
        st.error("Job_Machine_Quantities.xlsx not found!")

    # **Cache the file loading function to speed up performance**
    @st.cache_data
    def load_excel(file_path):
        return pd.read_excel(file_path)

    # Load Excel File (Cached for speed)
    file_path = "Job_Machine_Quantities.xlsx"
    df = load_excel(file_path)


    if df.shape[1] >= 8:
        # Extract Job Numbers from Column 1
        job_numbers = df.iloc[:, 0].dropna().unique().tolist()  

        # Create a selectbox for job numbers
        job_selection = st.selectbox("Select a Job Number", job_numbers, index=0)

        # Filter the dataframe based on selected job number
        filtered_df = df[df.iloc[:, 0] == job_selection]

        if not filtered_df.empty:
        # Read the Machine Input (Column 8) and Final Demand (Column 2)
            machine_input = filtered_df.iloc[:, 7].values  # Column index 8 (0-based index → 7)
            final_demand = filtered_df.iloc[:, 1].unique()  # Column index 2 (0-based index → 1)
            machine_name = filtered_df.iloc[:, 6].values  # Column index 7 (0-based index → 6)

            final_demand_val = float(final_demand[0]) 
            
            initial_input = float(machine_input[0])
            initial_input = round(initial_input, 3)

            # Display Results
            st.write(f"### Selected Job: {job_selection}")
            st.write(f"#### Starting Input: {initial_input}")

            col1,col2 = st.columns(2)
            with col1:
                st.write("**Machine Name:**")
                st.write(machine_name)
            with col2:
                st.write("**Machine Input:**")
                st.write(machine_input)

            st.write(f"#### Final Output: {final_demand_val}")
        else:
            st.error("No data found for the selected job number.")
    else:
        st.error("The uploaded file does not contain any columns.")

else:
    st.write("To view, import a file first.")