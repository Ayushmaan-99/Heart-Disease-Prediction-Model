print()  
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("Here's Disease Prediction by Ayushmaan🌑")
st.markdown('Enter the following details to predict the presence of heart disease:')

Age = st.slider('Age', 18, 100, 25)
Sex = st.selectbox('Sex',['male', 'female'])
Chestpain_type = st.selectbox('Chestpain_type', ['ATA', 'NAP', 'TA', 'ASY'])
Resting_bp = st.number_input('Resting_bp', 80,200,120)
Cholesterol = st.number_input('Cholesterol',100,600,200)
Fasting_bs = st.selectbox('Fasting blood sugar', [0,1])
Resting_ecg = st.selectbox('Resting_ecg', ['Normal', 'ST', 'LVH'])
Max_heart_rate = st.slider('Max_heart_rate', 60,220,150)
Excercise_angina = st.selectbox('Exercise_angina', ['Y','N'])
Oldpeak = st.slider('Oldpeak',0.0,10.0,1.0)
st_slop = st.selectbox('ST_slope', ['Up','Flat','Down'])


if st.button('Predict'):
   raw_data = {
       'Age': Age,
       'Resting_bp': Resting_bp,
        'Cholesterol': Cholesterol,
        'Fasting_bs': Fasting_bs,
        'Max_heart_rate': Max_heart_rate,
        'Oldpeak': Oldpeak,
        'Sex'+ Sex:1,
        'Chestpain_type' + Chestpain_type:1,
        'Resting_ecg' + Resting_ecg:1,  
        'Excercise_angina' + Excercise_angina:1,
        'ST_slope' + st_slop:1
    }
   
   input_df = pd.DataFrame([raw_data])
   for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
   input_df = input_df[columns]

   scaled_input = scaler.transform(input_df)
   Prediction = model.predict(scaled_input)[0]
   
   if Prediction == 1:
        st.error('⚠️ The person is likely to have heart disease. Please consult a doctor for further evaluation.')
   else:
        st.success('✅The person is unlikely to have heart disease. Maintain a healthy lifestyle!')
        

