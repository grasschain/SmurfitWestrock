import streamlit as st
import math

# Set page title
st.set_page_config(page_title="Single Job", layout="centered")

if st.button(label="Return to Home Page", key=None, help=None, type="secondary", icon=None,
            disabled=False, use_container_width=False):
    st.switch_page("pgTitle.py")

# Title
st.markdown("## Enter Job Information:")
    
left, middle, right = st.columns([3, 5, 8], vertical_alignment="top")

with left:
    orderQty = st.number_input("Order Qty", min_value=0)
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
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        feature1 = st.selectbox("OFFSET", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        feature2 = st.selectbox("FLUTE CODE", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        feature3 = st.selectbox("TEST CODE", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        feature4 = st.selectbox("CLOSURE TYPE", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        feature5 = st.selectbox("COMPONENT CODE", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        feature6 = st.selectbox("ROTARY DC", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
        feature7 = st.selectbox("NUMBER UP ENTRY", ["YES", "NO"],
                        index=None,
                        placeholder="Select")
    with col2:
        feature8 = st.number_input("BLANK WIDTH", min_value=0, value=None,
                        placeholder="Enter Value")
        feature9 = st.number_input("BLANK LENGTH", min_value=0, value=None,
                        placeholder="Enter Value")
        feature10 = st.number_input("ITEM WIDTH", min_value=0, value=None,
                        placeholder="Enter Value")
        feature11 = st.number_input("ITEM LENGTH", min_value=0, value=None,
                        placeholder="Enter Value")


st.markdown("---")  # Horizontal line

wasteQty = round(0.05 * orderQty)
optQty = round(orderQty - wasteQty)
finalQty = round(orderQty + (0.001 * orderQty))

#  Information Return
("# OPTIMAL STARTING QUANTITIES")

# Starting Quantity
st.write("### Start with: ", optQty)

# Estimated amount of waste
st.write("Estimated Waste: ", wasteQty)

# Finished Quantity
st.write("Final Output Quantity: ", finalQty)