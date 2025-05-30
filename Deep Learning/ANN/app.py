import streamlit as st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd 
import pickle 

# Get current script directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the trained model
model = tf.keras.models.load_model(os.path.join(dir_path, 'model.h5'))

# Load encoders and scaler
with open ('label_encoder_gender.pkl' , 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open ('ohe_geo.pkl' , 'rb') as file:
    ohe_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as f:
    scaler, expected_columns = pickle.load(f)
    
## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography' , ohe_geo.categories_[0])
gender = st.selectbox('Gender' , label_encoder_gender.classes_)
age = st.slider('Age' , 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure' , 0 , 10)
num_of_products = st.slider('Number of Products' , 1 ,4)
has_cr_card = st.selectbox('Has Credit Card' , [0,1])
is_active_member = st.selectbox('Is Active Member' , [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score] , 
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# One hot encode 'Geography'
geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns = ohe_geo.get_feature_names_out(['Geography']))

# Combine the data
input_data = pd.concat([input_data.reset_index(drop = True) , geo_encoded_df] , axis = 1)

# Scale the input data
input_data = input_data[expected_columns]
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f"Churn Probability: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
