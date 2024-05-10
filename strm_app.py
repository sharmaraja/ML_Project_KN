import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define function to set page configuration
def set_page_config():
    st.set_page_config(
        page_title="Student Performance Prediction",
        page_icon=":books:",
        layout="wide"
    )


# Define the Streamlit app
def main():

    # Define page background color
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f5f5f5;
        }
        </style>
        """,

        unsafe_allow_html=True
    )

    # Define page sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        'Select a page',
        ['Home', 'Predict Data']
    )


    st.title('Student Performance Prediction')

    # Render different pages based on user selection
    if page == 'Home':
        st.header('Welcome to Student Performance Prediction')
        st.write('This app allows you to predict student performance based on various factors.')
        st.image("rm347-porpla-02-a-01.jpg", use_column_width=True)  # Add image
        st.write('This app allows you to predict student performance based on various factors.')
        st.markdown("---")  # Add horizontal line
        st.subheader('Description:')
        st.write('Enter the information below to predict student performance.')
        st.markdown("---")  # Add horizontal line

    elif page == 'Predict Data':
        st.header('Predict Student Performance')
        st.write('Please provide the required information:')

        # Create form fields for user input
        gender = st.selectbox('Gender', ['male', 'female'])
        race_ethnicity = st.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
        parental_level_of_education = st.selectbox('Parental Education Level', ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"])
        lunch = st.selectbox('Lunch Type', ['standard', 'free/reduced'])
        test_preparation_course = st.selectbox('Test Preparation Course', ['none', 'completed'])
        reading_score = st.number_input('Reading Score', min_value=0, max_value=100, value=50)
        writing_score = st.number_input('Writing Score', min_value=0, max_value=100, value=50)

        # Handle prediction when user clicks the "Predict" button
        if st.button('Predict'):
            # Create CustomData object with user input
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            # Get prediction results using PredictPipeline
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Display prediction result
            st.subheader("Prediction for the student's maths score is: ")
            st.write(results[0])

# Run the Streamlit app
if __name__ == "__main__":
    main()
