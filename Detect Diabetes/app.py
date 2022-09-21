import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

model=load_model('rf')

st.title("app to check if a patient has diabetes")

# crete the frontend user interface
preg=st.number_input('Pregnancies', min_value=0)
glucose=st.slider('Glucose',min_value=0.0, max_value=200.0)
bp=st.slider('BloodPressure',min_value=0.0, max_value=200.0)
skin=st.slider('SkinThickness',min_value=0.0, max_value=20.0)
insulin=st.slider('Insulin',min_value=0.0, max_value=200.0)
bmi=st.slider('BMI',min_value=0.0, max_value=50.0)
dbf=st.slider('DiabetesPedigreeFunction',min_value=0.0 ,max_value=10.0)
age=st.slider('Age', min_value=20.0, max_value=100.0)


input_data={
    'Pregnancies':preg,
    'Glucose':glucose,
    'BloodPressure':bp,
    'SkinThickness':skin,
    'Insulin':insulin,
    'BMI':bmi,
    'DiabetesPedigreeFunction':dbf,
    'Age':age
    
}

input_data=pd.DataFrame([input_data])
st.write(input_data)

prediction=predict_model(model, input_data)
predicted_outcome=prediction['Label'][0]
output=str(predicted_outcome)
if st.button('Predict'):
    if output=='1':
        st.success('this person has diabetes')
    else:
        st.success("NO Diabetes")
