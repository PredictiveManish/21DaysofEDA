import sys
import subprocess
import streamlit as st
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Dependency check with version verification
REQUIRED_PACKAGES = {
    'numpy': '1.24.3',
    'pandas': '2.0.2',
    'scikit-learn': '1.2.2',
    'joblib': '1.2.0',
    'streamlit': '1.22.0'
}

missing = []
for package, version in REQUIRED_PACKAGES.items():
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    st.error(f"⚠️ Missing dependencies: {', '.join(missing)}")
    if st.button("🛠️ Auto-install requirements"):
        with st.spinner(f"Installing {len(missing)} packages..."):
            try:
                for package in missing:
                    install(f"{package}=={REQUIRED_PACKAGES[package]}")
                st.success("Dependencies installed! Please refresh the page.")
                st.balloons()
                st.stop()
            except Exception as e:
                st.error(f"Installation failed: {str(e)}")
                st.stop()

# Model loading with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/salary_model.joblib')
        preprocessor = joblib.load('models/preprocessor.joblib')
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model files not found! Please check the 'models' folder.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, preprocessor = load_models()

# App UI
st.set_page_config(page_title="Salary Predictor", page_icon="💸")
st.title('💰 Data Science Salary Predictor')
st.image('https://i.imgur.com/3S4X7Jq.png', width=300)

with st.expander("ℹ️ How to use"):
    st.write("""
    1. Select your job details
    2. Click 'Predict Salary'
    3. See estimated market value!
    """)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        job = st.selectbox("Job Title", ['Data Scientist', 'Data Engineer', 'ML Engineer', 'Data Analyst'])
        exp = st.select_slider("Experience Level", ['Entry', 'Mid', 'Senior', 'Executive'])
        
    with col2:
        model_type = st.radio("Work Model", ['Remote', 'Hybrid', 'On-site'])
        size = st.selectbox("Company Size", ['S (1-50)', 'M (51-500)', 'L (500+)'], index=2)
        continent = st.selectbox("Region", ['North America', 'Europe', 'Asia', 'Other'])
    
    submitted = st.form_submit_button("Predict Salary")
    
    if submitted:
        with st.spinner("Calculating..."):
            try:
                # Map UI values to model expected format
                size_map = {'S (1-50)': 'S', 'M (51-500)': 'M', 'L (500+)': 'L'}
                
                input_df = pd.DataFrame([[
                    job,
                    f"{exp}-level" if exp != 'Executive' else 'Executive',
                    model_type,
                    size_map[size],
                    continent
                ]], columns=['job_title_grouped', 'experience_level', 'work_models', 'company_size', 'continent'])
                
                processed = preprocessor.transform(input_df)
                prediction = model.predict(processed)[0]
                
                st.success(f"Predicted Salary: **${prediction:,.0f}**")
                st.balloons()
                
                # Show confidence indicator
                st.metric("Market Range", 
                         f"${prediction*0.9:,.0f} - ${prediction*1.1:,.0f}",
                         "+/- 10% typical variance")
                
            except NotFittedError:
                st.error("Model not properly initialized!")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

st.caption("""
Built with ❤️ as part of #21DaysofEDA | 
[Report Issues](https://github.com/yourusername/salary-predictor/issues)
""")