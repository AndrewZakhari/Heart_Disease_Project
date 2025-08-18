# ui/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the trained model ---
MODEL_PATH = 'models/final_model.pkl'

@st.cache_resource # Caches the model so it's not reloaded on every interaction
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        st.stop() # Stop the app if model is not found
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

model = load_model()

# --- Define feature names and input types ---
# IMPORTANT: This list MUST match the features used to train the final model
# and the order should match the training data if the model is sensitive to it.
# Based on the provided X_selected_features.csv:
FEATURE_NAMES = [
    'cp', 'chol', 'thalach', 'age', 'oldpeak', 'trestbps',
    'exang', 'thal', 'sex', 'restecg'
]

# Define possible categorical values for dropdowns/selections based on data description
CP_OPTIONS = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
SEX_OPTIONS = {0: "Female", 1: "Male"}
EXANG_OPTIONS = {0: "No", 1: "Yes"} # Exercise induced angina
THAL_OPTIONS = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"} # Check unique values in your data
RESTECG_OPTIONS = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}

# --- Streamlit App ---
def main():
    st.title("🫀 Heart Disease Prediction App")
    st.markdown("""
    This app predicts the likelihood of heart disease based on patient health data.
    Please enter the patient's information below.
    """)

    # --- Create Input Fields ---
    # Use Streamlit's layout features (columns) for a better UI
    col1, col2, col3 = st.columns(3)

    with col1:
        # Ensure the order and names match FEATURE_NAMES
        cp = st.selectbox("Chest Pain Type (cp)", options=list(CP_OPTIONS.keys()), format_func=lambda x: CP_OPTIONS[x])
        chol = st.number_input("Serum Cholesterol (chol) mg/dl", min_value=100.0, max_value=600.0, value=200.0, step=1.0)
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=50.0, max_value=250.0, value=150.0, step=1.0)

    with col2:
        age = st.number_input("Age", min_value=1.0, max_value=120.0, value=50.0, step=1.0)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        trestbps = st.number_input("Resting BP (trestbps) mmHg", min_value=50.0, max_value=250.0, value=120.0, step=1.0)

    with col3:
        exang = st.selectbox("Exercise Angina (exang)", options=list(EXANG_OPTIONS.keys()), format_func=lambda x: EXANG_OPTIONS[x])
        # Ensure thal options match your data's unique values (3, 6, 7)
        thal = st.selectbox("Thalassemia (thal)", options=list(THAL_OPTIONS.keys()), format_func=lambda x: THAL_OPTIONS[x])
        sex = st.selectbox("Sex", options=list(SEX_OPTIONS.keys()), format_func=lambda x: SEX_OPTIONS[x])
        restecg = st.selectbox("Resting ECG (restecg)", options=list(RESTECG_OPTIONS.keys()), format_func=lambda x: RESTECG_OPTIONS[x])

    # --- Prediction Button ---
    st.markdown("---") # Horizontal line
    if st.button("Predict Heart Disease Risk"):
        # --- Prepare Input Data ---
        # Create a dictionary with feature names and user inputs
        # The keys MUST exactly match the FEATURE_NAMES
        input_data = {
            'cp': cp,
            'chol': chol,
            'thalach': thalach,
            'age': age,
            'oldpeak': oldpeak,
            'trestbps': trestbps,
            'exang': exang,
            'thal': thal,
            'sex': sex,
            'restecg': restecg
        }

        # Convert to DataFrame (single row)
        # Crucially, reorder columns to match the FEATURE_NAMES order
        # This ensures the data fed to the model is in the correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_NAMES] # Reorder columns

        # Optional: Display the input DataFrame for debugging
        # st.write("Input DataFrame (for debugging):")
        # st.write(input_df)

        # --- Make Prediction ---
        try:
            # Check if model is loaded
            if model is None:
                 st.error("Model could not be loaded. Prediction cannot be performed.")
                 return

            # Get prediction (0 or 1)
            prediction = model.predict(input_df)[0]

            # Get prediction probability (for class 1 - presence of disease)
            # Ensure your model supports predict_proba
            prediction_proba = model.predict_proba(input_df)[0][1]

            # --- Display Results ---
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.error(f"⚠️ **High Risk** of Heart Disease (Probability: {prediction_proba:.2%})")
            else:
                st.success(f"✅ **Low Risk** of Heart Disease (Probability: {prediction_proba:.2%})")

            st.info("Note: This prediction is based on a machine learning model and should not be considered medical advice. Please consult a healthcare professional for a proper diagnosis.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check the input data and model compatibility.")

    # --- Optional: Add Information Section ---
    st.markdown("---")
    st.subheader("About the Features:")
    st.markdown("""
    *   **Chest Pain Type (cp):** 1 = Typical Angina, 2 = Atypical Angina, 3 = Non-anginal Pain, 4 = Asymptomatic
    *   **Serum Cholesterol (chol):** In mg/dl
    *   **Max Heart Rate (thalach):** Maximum heart rate achieved
    *   **Age:** Age in years
    *   **ST Depression (oldpeak):** Induced by exercise relative to rest
    *   **Resting BP (trestbps):** Resting blood pressure in mm Hg
    *   **Exercise Angina (exang):** Exercise induced angina (1 = Yes, 0 = No)
    *   **Thalassemia (thal):** 3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect
    *   **Sex:** 1 = Male, 0 = Female
    *   **Resting ECG (restecg):** 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy
    """)

if __name__ == "__main__":
    main()