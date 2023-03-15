import streamlit as st
import subprocess

# Install scikit-learn using pip
subprocess.call(['pip', 'install', 'pickle'])
import pickle
import pandas as pd
import numpy as np
import sklearn
import xgboost
from xgboost import XGBRegressor
import hashlib

pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['Australia',
 'India',
 'Bangladesh',
 'New Zealand',
 'South Africa',
 'England',
 'West Indies',
 'Afghanistan',
 'Pakistan',
 'Sri Lanka']

cities = ['Colombo',
 'Mirpur',
 'Johannesburg',
 'Dubai',
 'Auckland',
 'Cape Town',
 'London',
 'Pallekele',
 'Barbados',
 'Sydney',
 'Melbourne',
 'Durban',
 'St Lucia',
 'Wellington',
 'Lauderhill',
 'Hamilton',
 'Centurion',
 'Manchester',
 'Abu Dhabi',
 'Mumbai',
 'Nottingham',
 'Southampton',
 'Mount Maunganui',
 'Chittagong',
 'Kolkata',
 'Lahore',
 'Delhi',
 'Nagpur',
 'Chandigarh',
 'Adelaide',
 'Bangalore',
 'St Kitts',
 'Cardiff',
 'Christchurch',
 'Trinidad']

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)
def hash_string(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
with col1:
    batting_team = st.selectbox('Select batting team',sorted(teams))
    team1=hash_string(batting_team)
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))
    team2=hash_string(bowling_team)

city = st.selectbox('Select city',sorted(cities))
venue=hash_string(city)

col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'hashed_batting_team': [team1], 'hashed_bowling_team': [team2],'hashed_city':venue, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))



