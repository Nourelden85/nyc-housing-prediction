import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="NY House Price Predictor", layout="wide")

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'house_price_model.joblib')
    preprocessor_path = os.path.join(current_dir, 'models', 'house_preprocessor.joblib')
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

try:
    model, preprocessor = load_assets()
except:
    st.error("joblib files doesn't exist.")

def calculate_distance(lat1, lon1, lat2, lon2):
    p = np.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))

st.title("NY House Price Predictor 🏠🗽")
st.markdown("Locate the property on the map and enter the specifications for the forecast.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Location on the map 📍")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    
    folium.Marker([40.7580, -73.9855], popup="Times Square", icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([40.7060, -74.0088], popup="Wall Street", icon=folium.Icon(color="blue")).add_to(m)

    map_output = st_folium(m, height=400, width=500, returned_objects=[])
    
    selected_lat = 40.7128
    selected_lon = -74.0060
    
    if map_output['last_clicked']:
        selected_lat = map_output['last_clicked']['lat']
        selected_lon = map_output['last_clicked']['lng']
        st.success(f"Selected coordinates: {selected_lat:.4f}, {selected_lon:.4f}")

with col2:
    st.subheader("Property specifications 📝")
    prop_type = st.selectbox("Property type", ['Condo', 'House', 'Townhouse', 'Co-op', 'Multi-family home'])
    locality = st.selectbox("Locality", ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"])
    zipcode = st.number_input("Zipcode", value=10001, min_value=10001, max_value=11697)
    
    c1, c2, c3 = st.columns(3)
    beds = c1.number_input("Bedrooms", min_value=1, value=2)
    bath = c2.number_input("Bathrooms", min_value=1, value=2)
    sqft = c3.number_input("Sqft", min_value=100, value=1200)

    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", value=selected_lat, format="%.4f")
    with c2:
        lon = st.number_input("Longitude", value=selected_lon, format="%.4f")

if st.button("Predict the price 💰", use_container_width=True):
    dist_times = calculate_distance(selected_lat, selected_lon, 40.7580, -73.9855)
    dist_wall = calculate_distance(selected_lat, selected_lon, 40.7060, -74.0088)
    
    input_data = pd.DataFrame({
        'type': [prop_type],
        'locality': [locality],
        'zipcode': [zipcode],
        'beds': [beds],
        'bath': [bath],
        'propertysqft': [sqft],
        'distance_to_center': [dist_times],
        'distance_to_wall_street': [dist_wall]
    })
    
    try:
        transformed_X = preprocessor.transform(input_data)
        
        pred_log = model.predict(transformed_X)
        
        real_price = 10**pred_log[0]
        
        st.balloons()
        st.metric(label="Estimated price: ", value=f"${real_price:,.2f}")
        st.write(f"Distance to Times Square: {dist_times:.2f} km")
        st.write(f"Distance to Wall Street: {dist_wall:.2f} km")
        
    except Exception as e:
        st.error(f"Error: {e}")