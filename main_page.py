import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Function to preprocess data
def preprocess_data(df):
    numeric_features = ['age', 'signup_flow']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor.transform(df)

# Load your model
model_path = r'C:\Users\Public\Documents\model1.joblib'
model = load(model_path)

# Title for the app
st.title('Airbnb Booking Prediction')

# Input form for user query
st.subheader('Enter user details:')
age = st.number_input('Age', min_value=0, max_value=120)
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
# Add more input fields as needed

# Prepare user input for prediction
user_input = pd.DataFrame({'age': [age], 'gender': [gender]})  # Add more columns as needed
user_input_processed = preprocess_data(user_input)

# Make prediction
prediction = model.predict(user_input_processed)

# Display prediction
st.subheader('Prediction:')
st.write(f'The predicted country destination is: {prediction[0]}')