#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, ordinal_encoder
from load_model import get_model

model = get_model(model_path = r'Model/RTA_model.joblib')
#model = joblib.load(r"RTA_model.joblib")

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu

options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_sex = ['Male', 'Female', 'Unknown']

options_edu = ['Above high school', 'Junior high school', 
                                    'Elementary school' ,'High school', 'Unknown', 'Illiterate', 'Writing & reading']

options_exp = ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence', 'Below 1yr', 'unknown']

options_vehicle = ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)',  'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle']

options_vehicle_owner = ['Owner', 'Governmental','Organization', 'Other']

options_service_year = ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown', 'Below 1yr']   
    
options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

options_lanes = ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way', 'Two-way (divided with solid lines road marking)', 'Two-way (divided with broken lines road marking)']

options_allignment = ['Tangent road with flat terrain','Tangent road with mild grade and flat terrain', 'Escarpments',
                     'Tangent road with rolling terrain', 'Gentle horizontal curve', 'Tangent road with mountainous terrain and',
                     'Steep grade downward with mountainous terrain', 'Sharp reverse curve',
                     'Steep grade upward with mountainous terrain']

options_junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape']

options_surface_type = ['Asphalt roads', 'Earth roads','Asphalt roads with some distress', 'Gravel roads' 'Other']
    
options_surface_conditions = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']

options_light = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit']

options_weather = ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Unknown', 'Fog or mist']

options_collision = ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
                    'Collision with roadside objects', 'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles',
                    'Collision with pedestrians', 'With Train', 'Unknown']
    


options_vehicle_movement = ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go', 'Getting off',
                            'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking', 'Other', 'Entering a junction']

options_casualty_class = ['Driver or rider', 'Pedestrian' 'Passenger']

options_casualty_sex = ['Male', 'Female']

options_casualty_age = ['31-50', '18-30', 'Under 18', 'Over 51']

options_casualty_severity = ['3', '2', '1']
    
options_pedestrian_movement = ['Not a Pedestrian', "Crossing from driver's nearside",
                              'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
                              'Unknown or other',
                              'Crossing from offside - masked by parked or statioNot a Pedestrianry vehicle',
                              'In carriageway, statioNot a Pedestrianry - not crossing (standing or playing)',
                              'Walking along in carriageway, back to traffic',
                              'Walking along in carriageway, facing traffic',
                              'In carriageway, statioNot a Pedestrianry - not crossing (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle']


options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']




features = ['day_of_week', 'driver_age', 'driver_sex', 'educational_level',
       'driving_experience', 'vehicle_type', 'vehicle_owner', 'service_year',
       'accident_area', 'lanes', 'road_allignment', 'junction_type',
       'surface_type', 'road_surface_conditions', 'light_condition',
       'weather_condition', 'collision_type', 'vehicles_involved',
       'casualties', 'vehicle_movement', 'casualty_class', 'casualty_sex',
       'casualty_age', 'casualty_severity', 'pedestrian_movement',
       'accident_cause', 'hour', 'minute']

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        vehicles_involved = st.slider("Select number of vehicles involved: ", 1, 7, value=0, format="%d")
        casualties = st.slider("Select number of causalties involved: ", 1, 8, value=0, format="%d")
        hour = st.slider("Select hour of accident: ", 0, 23, value=0, format="%d")
        minute = st.slider("Approx. Pickup Minute: ", 0, 59, value=0, format="%d")
        day_of_week = st.selectbox("Select day of the week: ", options=options_day)
        driver_age = st.selectbox("Select driver age: ", options=options_age)
        driver_sex = st.selectbox("Select driver sex: ", options=options_sex)
        education =  st.selectbox("Select driver ecucational level: ", options=options_edu)
        driver_experience = st.selectbox("Select driver experience: ", options=options_exp)
        vehicle_type = st.selectbox("Select vehicle type : ", options=options_vehicle)
        vehicle_owner = st.selectbox("Select vehicle owner : ", options=options_vehicle_owner)
        service_year = st.selectbox("Select vehicle service period : ", options=options_service_year)
        accident_area = st.selectbox("Select accident area: ", options=options_acc_area)
        lanes = st.selectbox("Select lane: ", options=options_lanes)
        allignment = st.selectbox("Select road allignment: ", options=options_allignment)
        junction = st.selectbox("Select junction type: ", options=options_junction)
        surface_type = st.selectbox("Select road surface type: ", options=options_surface_type)
        surface_conditions = st.selectbox("Select road surface condition: ", options=options_surface_conditions)
        light = st.selectbox("Select light condition: ", options=options_light)
        weather = st.selectbox("Select weather condition: ", options=options_weather)
        collision = st.selectbox("Select collision type: ", options=options_collision)
        vehicle_movement = st.selectbox("Select vehicle movement: ", options=options_vehicle_movement)
        casualty_class = st.selectbox("Select causalty class: ", options=options_casualty_class)
        casualty_sex = st.selectbox("Select casualty sex: ", options=options_casualty_sex)
        casualty_age = st.selectbox("Select casualty age: ", options=options_casualty_age)
        casualty_severity = st.selectbox("Select casualty severity: ", options=options_casualty_severity)
        pedestrian_movement = st.selectbox("Select pedestrian movement: ", options=options_pedestrian_movement)
        cause = st.selectbox("Select accident cause: ", options=options_cause)
        
  
        
        submit = st.form_submit_button("Predict")


    if submit:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        driver_age = ordinal_encoder(driver_age, options_age)
        driver_sex = ordinal_encoder(driver_sex, options_sex)
        education =  ordinal_encoder(education, options_edu)
        driver_experience = ordinal_encoder(driver_experience, options_exp)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle)
        vehicle_owner = ordinal_encoder(vehicle_owner, options_vehicle_owner)
        service_year = ordinal_encoder(service_year, options_service_year)
        accident_area = ordinal_encoder(accident_area, options_acc_area)
        lanes = ordinal_encoder(lanes, options_lanes)
        allignment = ordinal_encoder(allignment, options_allignment)
        junction = ordinal_encoder(junction, options_junction)
        surface_type = ordinal_encoder(surface_type, options_surface_type)
        surface_conditions = ordinal_encoder(surface_conditions, options_surface_conditions)
        light = ordinal_encoder(light, options_light)
        weather = ordinal_encoder(weather, options_weather)
        collision = ordinal_encoder(collision, options_collision)
        vehicle_movement = ordinal_encoder(vehicle_movement, options_vehicle_movement)
        casualty_class = ordinal_encoder(casualty_class, options_casualty_class)
        casualty_sex = ordinal_encoder(casualty_sex, options_casualty_sex)
        casualty_age = ordinal_encoder(casualty_age, options_casualty_age)
        casualty_severity = ordinal_encoder(casualty_severity, options_casualty_severity)
        pedestrian_movement = ordinal_encoder(pedestrian_movement, options_pedestrian_movement)
        cause = ordinal_encoder(cause, options_cause)
        
        

        data = np.array([vehicles_involved, casualties, hour, minute, day_of_week, driver_age, driver_sex, education,
                         driver_experience, vehicle_type, vehicle_owner, service_year, accident_area, lanes, allignment, 
                         junction, surface_type, surface_conditions, light, weather, collision, vehicle_movement,casualty_class,
                         casualty_sex, casualty_age, casualty_severity, pedestrian_movement, cause]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)
        
        if pred[0]==1:
              x='slight injury'
        elif pred[0]==2:
              x='serious injury'
        else:
              x='fatal injury'

        st.write(f"The predicted accident severity is:  {x}")

if __name__ == '__main__':
    main()

