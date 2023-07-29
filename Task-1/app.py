import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Function to load the trained model
def load_best_model():
    # Load the trained model from the pickle file saved in Model.ipynb
    model = joblib.load('model.pickle')
    return model

# Function to load the scaler
def load_scaler():
    # Load the scaler from the pickle file saved in EDA.ipynb
    scaler = joblib.load('scaler.pickle')
    return scaler

# Function to make prediction
def predict_admission_chance(best_model, scaler, input_data):
    # Scale the user input using the same scaler used for training data
    user_input_scaled = scaler.transform(input_data)
    
    # Make prediction
    admission_chance = best_model.predict(user_input_scaled)[0]
    return admission_chance

# Streamlit app code 
def main():
    # Add image and title
    st.image("X:\Internship\INTERNSAVY-s_TASK\Task-1\graduation-ceremony.png", use_column_width=True, output_format="auto", caption='image')
    st.markdown('''
                # Graduate School Admission Chance Predictor 🎓
                ''')
    
    # Sidebar for user input
    st.sidebar.title("Please provide your inputs:")
    gre_score = st.sidebar.slider("GRE Score", min_value=260, max_value=340, value=320)
    toefl_score = st.sidebar.slider("TOEFL Score", min_value=80, max_value=120, value=110)
    university_rating = st.sidebar.slider("University Rating", min_value=1, max_value=5, value=4)
    sop_score = st.sidebar.slider("Statement of Purpose (SOP) Score", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    lor_score = st.sidebar.slider("Letter of Recommendation (LOR) Score", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    cgpa = st.sidebar.slider("CGPA", min_value=6.0, max_value=10.0, value=8.8, step=0.1)
    research = st.sidebar.radio("Research (0 for No, 1 for Yes)", options=[0, 1], index=1)

    user_input_df = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [university_rating],
        'SOP': [sop_score],
        'LOR': [lor_score],
        'CGPA': [cgpa],
        'Research': [research]
    })

    if st.sidebar.button("Predict"):
        # Load the best model and scaler (replace with your model and scaler)
        best_model = load_best_model()  # Implement the function to load your model
        scaler = load_scaler()          # Implement the function to load your scaler

        # Add progress bar
        with st.spinner("Predicting..."):
            prediction_progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate prediction time
                prediction_progress.progress(i + 1)

        # Make prediction using the model
        admission_chance = predict_admission_chance(best_model, scaler, user_input_df)

        # Display the result
        st.success(f"\nPredicted Chance of Admit: {admission_chance:.4f}")

    # Display user-inputted values as tabular data in the center
    st.write("\n\n**User-Inputted Values**")
    st.dataframe(user_input_df)

if __name__ == "__main__":
    main()
