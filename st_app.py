import streamlit as st 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle 

#load model
try:
    model=tf.keras.models.load_model('model.h5')
except OSError as e: 
    st.write(f"Error opening model: {e}")
    # print(f"Error opening model: {e}")

## load the encoder and scaler
with open('OHE_geo.pkl','rb') as file:
    OHE_geo=pickle.load(file)

with open('labelencoder_Gender.pkl','rb') as file:
    labelencoder_Gender=pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)

#Streamlit app

st.title('Customer churn prediction')

#user input

geography = st.selectbox('Geography', OHE_geo.categories_[0])
gender = st.selectbox('Gender', labelencoder_Gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [labelencoder_Gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# One-hot encode 'Geography'
geo_encoded = OHE_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=OHE_geo.get_feature_names_out())

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scalar.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')