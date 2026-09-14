import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️")
st.title("✈️ Flight Price Predictor")
st.write("Enter the flight details and predict the ticket price.")

model = joblib.load("flight_price_model.pkl")

# Values from the uploaded Clean_Dataset.csv
AIRLINES = ["AirAsia", "Air India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
TIMES = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS = ["zero", "one", "two_or_more"]
CLASSES = ["Economy", "Business"]

hour_map = {
    "Early_Morning": 6,
    "Morning": 9,
    "Afternoon": 14,
    "Evening": 18,
    "Night": 22,
    "Late_Night": 2,
}

airline = st.selectbox("Airline", AIRLINES)
source_city = st.selectbox("Source City", CITIES)
destination_city = st.selectbox("Destination City", CITIES)
departure_time = st.selectbox("Departure Time", TIMES)
arrival_time = st.selectbox("Arrival Time", TIMES)
stops = st.selectbox("Stops", STOPS)
flight_class = st.selectbox("Class", CLASSES)

duration = st.number_input("Duration (hours)", min_value=0.1, max_value=50.0, value=2.5, step=0.1)
days_left = st.number_input("Days Left", min_value=1, max_value=50, value=15, step=1)

if st.button("Predict Price"):
    user_input = pd.DataFrame([{
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": flight_class,
        "duration": duration,
        "days_left": days_left,
        "departure_hour": hour_map[departure_time],
        "arrival_hour": hour_map[arrival_time],
    }])

    prediction = model.predict(user_input)[0]
    st.success(f"Estimated Flight Price: ₹{prediction:,.0f}")
