import streamlit as st
import numpy as np
import joblib


st.set_page_config(
    page_title="Coffee Sales Prediction",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Coffee Sales Prediction App")
st.write("Predict coffee sales using Machine Learning")


@st.cache_resource
def load_artifacts():
    model = joblib.load("coffee_sales_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("feature_selector.pkl")
    return model, scaler, selector

model, scaler, selector = load_artifacts()

st.subheader("📥 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    Temperature_C = st.number_input("Temperature (°C)", value=25.0,step=1.0)
    Is_Weekend = st.selectbox("Is Weekend?", [0, 1])
    Is_Raining = st.selectbox("Is Raining?", [0, 1])

with col2:
    Num_Customers = st.number_input("Number of Customers", value=100, step=1)
    Staff_Count = st.number_input("Staff Count", value=5, step=1)
    Promotion_Active = st.selectbox("Promotion Active?", [0, 1])


if st.button("🚀 Predict Sales"):
    try:
        input_data = np.array([[
            Temperature_C,
            Is_Weekend,
            Is_Raining,
            Num_Customers,
            Staff_Count,
            Promotion_Active
        ]])

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        st.success(f"📈 Predicted Coffee Sales: **{prediction[0]:.2f} units**")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
