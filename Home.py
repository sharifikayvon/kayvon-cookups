import streamlit as st

st.set_page_config(
    page_title="kayvon-cookups",
    page_icon="👨‍🍳",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<h1 style='text-align: center'>Kayvon's Web App Repository</h1>",
    unsafe_allow_html=True,
)

st.markdown("""
Welcome to my collection of web apps 🤠 !

Features include:
- **Graph and Fit your Data**: Make a scatter plot of your data, specify a model, and fit.
- **Photos to Spectra**: Upload a photo and see it reimagined as a spectrum of light.
- **Visualize 1D Motion**: Define **x(t)**, **v(t)**, or **a(t)** and visualize all three simultaneously.

Explore the tools using the sidebar on the left. 

Please email me at ksharifi1@gsu.edu with any questions, feedback, or especially web app ideas!

""")
