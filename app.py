import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(page_title="OAK Breast Cancer Predictor", layout="wide")

@st.cache_resource
def train_model():
    # Load the data from sklearn
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
    data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
    data_frame['label'] = breast_cancer_dataset.target
    
    # Separating features and target
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']
    
    # Splitting the data (using random_state=2 as per your notebook)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    # Model training
    model = LogisticRegression(max_iter=5000) # Increased max_iter to handle convergence
    model.fit(X_train, Y_train)
    
    return model, breast_cancer_dataset.feature_names

# Initialize model and features
model, feature_names = train_model()

# App UI
st.title("Breast Cancer Prediction System")
st.markdown("""
This application uses a **Logistic Regression** model trained on the Scikit-Learn Breast Cancer dataset to predict tumor types.
""")

st.sidebar.header("Input Tumor Measurements")
st.sidebar.info("Adjust the values below to get a prediction.")

# Create input fields for all 30 features
def get_user_inputs():
    inputs = []
    # We create two columns in the sidebar for better spacing
    col1, col2 = st.sidebar.columns(2)
    
    for i, name in enumerate(feature_names):
        # Determine which column to place the input in
        target_col = col1 if i % 2 == 0 else col2
        # Using a number input for precision
        val = target_col.number_input(f"{name}", value=float(np.round(model.coef_[0][i] * 10, 2) if i < len(model.coef_[0]) else 1.0))
        inputs.append(val)
    return np.array(inputs).reshape(1, -1)

input_df = get_user_inputs()

# Prediction Logic
st.subheader("Results")
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    col_res, col_prob = st.columns(2)
    
    with col_res:
        if prediction[0] == 0:
            st.error("### Prediction: Malignant")
            st.write("The model suggests the tumor is cancerous.")
        else:
            st.success("### Prediction: Benign")
            st.write("The model suggests the tumor is non-cancerous.")
            
    with col_prob:
        st.write("#### Confidence Levels")
        st.write(f"Malignant Probability: **{prediction_proba[0][0]*100:.2f}%**")
        st.write(f"Benign Probability: **{prediction_proba[0][1]*100:.2f}%**")

# Model Metadata (Optional)
with st.expander("View Model Training Details"):
    st.write("This model was trained with an 80/20 split on 569 samples.")
    st.write("Target Distribution: 357 Benign (1), 212 Malignant (0).")
