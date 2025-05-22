import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load your data
df = pd.read_csv('data_science_salaries.csv')

# Continent mapping
continent_mapping = {
    'United States': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',
    'United Kingdom': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Poland': 'Europe',
    'Spain': 'Europe',
    'Netherlands': 'Europe',
    'Switzerland': 'Europe',
    'India': 'Asia',
    'China': 'Asia',
    'Japan': 'Asia',
    'Singapore': 'Asia',
    'Australia': 'Oceania',
    'New Zealand': 'Oceania',
    'Brazil': 'South America',
    'Argentina': 'South America',
    'South Africa': 'Africa',
    'United Arab Emirates': 'Middle East',
    'Israel': 'Middle East'
}
# Apply continent mapping
df['continent'] = df['company_location'].map(continent_mapping)
df['continent'] = df['continent'].fillna('Other')
job_title_mapping = {
    'Data Scientist': [
        'Data Scientist', 'Applied Data Scientist', 'Data Science Practitioner',
        'Staff Data Scientist', 'Marketing Data Scientist', 'Data Science Engineer'
    ],
    'Data Analyst': [
        'Data Analyst', 'Business Data Analyst', 'Product Data Analyst',
        'Marketing Data Analyst', 'Finance Data Analyst', 'Compliance Data Analyst',
        'Financial Data Analyst', 'Staff Data Analyst', 'Insight Analyst'
    ],
    'Machine Learning Engineer': [
        'Machine Learning Engineer', 'ML Engineer', 'Machine Learning Software Engineer',
        'Machine Learning Researcher', 'Machine Learning Developer',
        'Machine Learning Scientist', 'Machine Learning Specialist',
        'Machine Learning Manager', 'Machine Learning Modeler',
        'Principal Machine Learning Engineer', 'Staff Machine Learning Engineer',
        'Head of Machine Learning', 'Lead Machine Learning Engineer',
        'Machine Learning Operations Engineer', 'Machine Learning Research Engineer'
    ],
    'BI Analyst': [
        'Business Intelligence Analyst', 'BI Analyst', 'BI Data Analyst',
        'BI Developer', 'BI Data Engineer', 'Business Intelligence Developer',
        'Business Intelligence Data Analyst', 'Business Intelligence Specialist',
        'Business Intelligence Manager'
    ],
    'Data Engineer': [
        'Data Engineer', 'Cloud Data Engineer', 'Software Data Engineer',
        'Data Infrastructure Engineer', 'Lead Data Engineer', 'ETL Developer',
        'ETL Engineer', 'Cloud Database Engineer', 'AWS Data Architect',
        'Azure Data Engineer', 'Consultant Data Engineer', 'Principal Data Engineer'
    ],
    'Data Architect': [
        'Data Architect', 'AI Architect', 'Big Data Architect', 'Cloud Data Architect',
        'Principal Data Architect'
    ],
    'Data Manager': [
        'Data Manager', 'Data Product Manager', 'Data Operations Manager',
        'Data Strategy Manager', 'Manager Data Management', 'Head of Data',
        'Director of Data Science', 'Managing Director Data Science'
    ],
    'Data Scientist Manager': [
        'Data Science Manager', 'Head of Data Science', 'Lead Data Scientist',
        'Data Science Lead', 'Principal Data Scientist', 'Data Science Director',
        'Data Science Tech Lead'
    ],
    'Researcher': [
        'Research Scientist', 'Research Engineer', 'AI Research Engineer',
        'Deep Learning Engineer', 'Deep Learning Researcher', 'Applied Machine Learning Scientist',
        'AI Scientist'
    ],
    'Others': []  # Everything not explicitly mapped
}

def map_job_title(original_title):
    for group, titles in job_title_mapping.items():
        if original_title in titles:
            return group
    return 'Others'

df['job_title_grouped'] = df['job_title'].apply(map_job_title)


# 1. Select features and target
features = ['job_title_grouped', 'experience_level', 'work_models', 'company_size', 'continent']
X = df[features]
y = df['salary_in_usd']

# 2. Preprocessing - OneHotEncode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ],
    remainder='passthrough'
)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Fit preprocessor and transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# 6. Save both model and preprocessor
joblib.dump(model, 'salary_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')

# Prediction function
def predict_salary(job_title, experience, work_model, company_size, continent):
    # Load model and preprocessor
    model = joblib.load('salary_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    
    # Create input DataFrame
    input_data = pd.DataFrame([[
        job_title,
        experience,
        work_model,
        company_size,
        continent
    ]], columns=features)
    
    # Preprocess and predict
    processed_data = preprocessor.transform(input_data)
    return model.predict(processed_data)[0]

# Example prediction
try:
    predicted = predict_salary(
        job_title='Data Scientist',
        experience='Senior-level',
        work_model='Remote',
        company_size='Large',
        continent='North America'
    )
    print(f"Predicted salary: ${predicted:,.2f}")
except Exception as e:
    print(f"Prediction error: {str(e)}")