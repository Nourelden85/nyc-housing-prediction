import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import folium_static

st.set_page_config(page_title="NY House Price Predictor", layout="wide")

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'house_price_model.joblib')
    preprocessor_path = os.path.join(current_dir, 'models', 'house_preprocessor.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

model, preprocessor = load_assets()

def calculate_distance(lat1, lon1, lat2, lon2):
    p = np.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))

st.title("NY House Price Predictor 🏠🗽")
st.markdown("Enter the property details and location to get a price estimate.")

if model is None:
    st.error("🚨 Model files not found! Please ensure 'house_price_model.joblib' and 'house_preprocessor.joblib' are in the 'models' folder.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Location Settings 📍")
        lat_input = st.number_input("Property Latitude", value=40.7128, format="%.4f")
        lon_input = st.number_input("Property Longitude", value=-74.0060, format="%.4f")
        
        m = folium.Map(location=[lat_input, lon_input], zoom_start=12)
        
        folium.Marker([40.7580, -73.9855], popup="Times Square", icon=folium.Icon(color="red")).add_to(m)
        folium.Marker([40.7060, -74.0088], popup="Wall Street", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker([lat_input, lon_input], popup="Selected House", icon=folium.Icon(color="green")).add_to(m)

        folium_static(m, height=400, width=550)

    with col2:
        st.subheader("Property Specifications 📝")
        prop_type = st.selectbox("Property Type", ['Condo', 'House', 'Townhouse', 'Co-op', 'Multi-family home'])
        locality = st.selectbox("Locality", ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"])
        zipcode = st.number_input("Zipcode", value=10001, min_value=10001, max_value=11697)
        
        c1, c2, c3 = st.columns(3)
        beds = c1.number_input("Bedrooms", min_value=0, value=2)
        bath = c2.number_input("Bathrooms", min_value=0, value=2)
        sqft = c3.number_input("Square Feet", min_value=100, value=1200)

    if st.button("Predict the Price 💰", use_container_width=True):
        dist_times = calculate_distance(lat_input, lon_input, 40.7580, -73.9855)
        dist_wall = calculate_distance(lat_input, lon_input, 40.7060, -74.0088)
        
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
            
            st.divider()
            st.balloons()
            st.metric(label="Estimated Market Value", value=f"${real_price:,.2f}")
            
            col_a, col_b = st.columns(2)
            col_a.info(f"📍 Distance to Times Square: {dist_times:.2f} km")
            col_b.info(f"📍 Distance to Wall Street: {dist_wall:.2f} km")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")