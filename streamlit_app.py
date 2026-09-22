import streamlit as st

st.set_page_config(page_title="phys1111toolkit", page_icon="🍎", layout="wide")
st.markdown("<h1 style='text-align: center'>phys1111 Data Visualization and Analysis Toolkit</h1>", unsafe_allow_html=True)

st.markdown(
    """
Welcome to the **phys1111 Data Analysis Toolkit**! This website was designed to 
streamline routine in-class data visualization and analysis tasks for students in 
Dr. Barooni's Physics 1111 course.

Features include:
- **Graph and Fit your Data**: Make a scatter plot of your data and fit a linear or quadratic function.
- **Photos to Spectra**: Upload a photo and see it reimagined as a spectrum of light.
- **Visualize 1D Motion**: Define **x(t)**, **v(t)**, or **a(t)** and visualize all three simultaneously.

Explore the tools using the sidebar on the left. If you have any questions or feedback, please email me at ksharifi1@gsu.edu.

More features to be added soon.


"""
)
