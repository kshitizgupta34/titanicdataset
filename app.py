import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.title("🚢 Titanic Survival Prediction")

model_titanic = pickle.load(open('model_titanic.pkl','rb'))

# User input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.1)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

sex_male = 1 if sex == "Male" else 0  # Model was trained with 'Sex_male'
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

user_input={'Pclass':pclass,
    'Age':age,
   'SibSp':sibsp,
   'Parch':parch,
   'Fare':fare,
   'Sex_male': sex_male,
   'Embarked_Q':embarked_Q,
   'Embarked_S':embarked_S
   }

user_input_df=pd.DataFrame(user_input,index=[0])

# Predict button
if st.button("Predict Survival"):
    prediction = model_titanic.predict(user_input_df)
    
    # Display result
    result = "Survived 🟢" if prediction[0] == 1 else "Did Not Survive 🔴"
    st.success(f"Prediction: **{result}**")
