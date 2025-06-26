import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
import io
import sys
from matplotlib.pyplot import text

# Constants for consistent page sizing
PAGE_WIDTH = 8.5  # inches
PAGE_HEIGHT = 11   # inches
DPI = 300          # resolution

# Redirect print statements to capture them
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout

# Create PDF file
pdf_pages = PdfPages('Data_Science_Analysis.pdf')

# Load your dataset
df = pd.read_csv('data_science_salaries.csv')

# Job title mapping (your existing code)
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

# Function to add a page with text
def add_text_page(text_content, title=None):
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
    fig.patch.set_visible(False)
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    ax.text(0.05, 0.95, text_content, 
            ha='left', va='top', wrap=True, 
            fontsize=10, transform=ax.transAxes,
            bbox=dict(facecolor='none', edgecolor='none', pad=10.0))
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close()

# Function to save plot to PDF with consistent sizing
def save_plot_to_pdf(fig=None, title=None):
    if fig is None:
        fig = plt.gcf()
    
    # Set the figure size if not already set
    if fig.get_size_inches()[0] != PAGE_WIDTH or fig.get_size_inches()[1] != PAGE_HEIGHT:
        fig.set_size_inches(PAGE_WIDTH, PAGE_HEIGHT)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_pages.savefig(fig, bbox_inches='tight', dpi=DPI)
    plt.close()

# 1. Basic Info Page ========================================================
basic_info = f"""
Dataset Information:
- Number of rows: {len(df)}
- Number of columns: {len(df.columns)}
- Columns: {', '.join(df.columns)}

First 5 rows:
{df.head().to_string()}

Data Types:
{df.dtypes.to_string()}
"""
add_text_page(basic_info, "Dataset Overview")

# 1. Pattern Discovery ======================================================

# Trend in salaries over work_year
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
sns.lineplot(data=df, x='work_year', y='salary_in_usd', estimator='median', errorbar=None)
plt.title('Median Salary Trend Over Years', pad=20)
save_plot_to_pdf()

# Remote jobs becoming more common
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
remote_trend = df.groupby('work_year')['work_models'].value_counts(normalize=True).unstack()
remote_trend.plot(kind='bar', stacked=True)
plt.title('Remote Work Trend Over Years', pad=20)
plt.ylabel('Proportion')
save_plot_to_pdf()

# Salary growth with experience level by job role
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', hue='job_title_grouped')
plt.title('Salary Distribution by Experience Level and Job Role', pad=20)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
save_plot_to_pdf()

# 2. Storytelling with Contrasts ============================================

# Compare two similar job roles (Data Analyst vs Data Scientist)
da_ds = df[df['job_title_grouped'].isin(['Data Analyst', 'Data Scientist'])]

# Salary comparison
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
sns.boxplot(data=da_ds, x='job_title_grouped', y='salary_in_usd')
plt.title('Salary Comparison: Data Analyst vs Data Scientist', pad=20)
save_plot_to_pdf()

# Work model preference
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
pd.crosstab(da_ds['job_title_grouped'], da_ds['work_models']).plot(kind='bar')
plt.title('Work Model Preference: Data Analyst vs Data Scientist', pad=20)
save_plot_to_pdf()

# Most common locations
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
da_ds.groupby('job_title_grouped')['company_location'].value_counts().unstack().plot(kind='bar', stacked=True)
plt.title('Job Title Distribution by Location', pad=20)
save_plot_to_pdf()

# Compare two countries/regions (US vs Poland in this example)
countries = df[df['company_location'].isin(['United States', 'Poland'])]

# Salary comparison
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
sns.boxplot(data=countries, x='company_location', y='salary_in_usd')
plt.title('Salary Comparison: US vs Poland', pad=20)
save_plot_to_pdf()

PAGE_WIDTH = 11  # Landscape width in inches
PAGE_HEIGHT = 8.5 # Landscape height in inches
DPI = 300         # resolution

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

# Then apply your job title grouping
df['job_title_grouped'] = df['job_title'].apply(map_job_title)

# Now create the filtered dataframe for comparison
da_ds = df[df['job_title_grouped'].isin(['Data Analyst', 'Data Scientist'])]

# Then proceed with the continental analysis
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
continent_data = da_ds.groupby(['continent', 'job_title_grouped']).size().unstack()

# Calculate percentages
continent_pct = continent_data.div(continent_data.sum(axis=1), axis=0) * 100

continent_pct.plot(kind='barh', stacked=True, width=0.85)
plt.title('Job Title Distribution by Continent (Percentage)', pad=20, fontsize=14)
plt.xlabel('Percentage (%)')
plt.ylabel('Continent')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Job Title')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlim(0, 100)
save_plot_to_pdf()

# 3. Anomaly & Outlier Investigation =======================================

# Detect salary outliers using IQR
Q1 = df['salary_in_usd'].quantile(0.25)
Q3 = df['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['salary_in_usd'] < lower_bound) | (df['salary_in_usd'] > upper_bound)]
outliers_text = "Outliers detected by IQR method:\n" + outliers_iqr[['job_title_grouped', 'experience_level', 'company_location', 'salary_in_usd']].to_string()

# Detect outliers using Z-score
# Detect outliers using Z-score
df['z_score'] = stats.zscore(df['salary_in_usd'])
outliers_z = df[np.abs(df['z_score']) > 3]

# Use continent instead of company_location in the output
outliers_text1 = "Outliers detected by IQR method:\n" + outliers_iqr[['job_title_grouped', 'experience_level', 'continent', 'salary_in_usd']].to_string()
outliers_text2 = "\n\nOutliers detected by Z-score method:\n" + outliers_z[['job_title_grouped', 'experience_level', 'continent', 'salary_in_usd']].to_string()
add_text_page(outliers_text2, "Outlier Detection")
add_text_page(outliers_text1, "Outlier Detection")

# Outlier visualization by continent
if not outliers_iqr.empty:
    plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    
    # Create the plot
    ax = sns.scatterplot(
        data=df, 
        x='continent',  # Changed from company_location to continent
        y='salary_in_usd', 
        hue='job_title_grouped',
        palette='tab20',  # More distinct colors
        alpha=0.7,       # Slightly transparent points
        s=100           # Larger points
    )
    
    # Customize the plot
    plt.title('Salary Outliers by Continent and Job Title', pad=20, fontsize=14)
    plt.xlabel('Continent', fontsize=12)
    plt.ylabel('Salary in USD', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Better angled labels
    
    # Highlight the outliers
    if not outliers_z.empty:
        outlier_points = ax.scatter(
            x=outliers_z['continent'],
            y=outliers_z['salary_in_usd'],
            s=150,                      # Even larger for outliers
            edgecolor='red',
            facecolor='none',
            linewidth=2,
            label='Statistical Outliers (z > 3)'
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    save_plot_to_pdf()
# 4. Insight-Driven Questions ==============================================

# Which roles offer the highest salary per level of experience?
salary_by_exp_role = df.groupby(['experience_level', 'job_title_grouped'])['salary_in_usd'].median().unstack()
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
salary_by_exp_role.plot(kind='bar')
plt.title('Median Salary by Experience Level and Job Title', pad=20)
plt.ylabel('Salary in USD')
save_plot_to_pdf()

# Company size vs salary
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
sns.boxplot(data=df, x='company_size', y='salary_in_usd')
plt.title('Salary Distribution by Company Size', pad=20)
save_plot_to_pdf()

# Remote work pay comparison
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
remote_pay = df.groupby('work_models')['salary_in_usd'].median()
remote_pay.plot(kind='bar')
plt.title('Median Salary by Work Model', pad=20)
plt.ylabel('Salary in USD')
save_plot_to_pdf()

# Location match effect
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
df['location_match'] = df['company_location'] == df['employee_residence']
location_effect = df.groupby('location_match')['salary_in_usd'].median()
location_effect.plot(kind='bar')
plt.title('Salary Comparison: Matching vs Different Locations', pad=20)
plt.ylabel('Median Salary in USD')
plt.xticks([0, 1], ['Different', 'Matching'], rotation=0)
save_plot_to_pdf()

# 5. Salary Bands/Bins ====================================================

# Create salary bins
bins = [0, 50000, 150000, 300000, float('inf')]
labels = ['Low (<$50k)', 'Mid ($50k-$150k)', 'High ($150k-$300k)', 'Elite ($300k+)']
df['salary_band'] = pd.cut(df['salary_in_usd'], bins=bins, labels=labels)

# Job titles per bin
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
pd.crosstab(df['job_title_grouped'], df['salary_band']).plot(kind='bar', stacked=True)
plt.title('Job Title Distribution Across Salary Bands', pad=20)
plt.ylabel('Count')
plt.legend(title='Salary Band', bbox_to_anchor=(1.05, 1), loc='upper left')
save_plot_to_pdf()

# Experience level per bin
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
pd.crosstab(df['experience_level'], df['salary_band']).plot(kind='bar', stacked=True)
plt.title('Experience Level Distribution Across Salary Bands', pad=20)
plt.ylabel('Count')
save_plot_to_pdf()

# Company sizes per bin
plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
pd.crosstab(df['company_size'], df['salary_band']).plot(kind='bar', stacked=True)
plt.title('Company Size Distribution Across Salary Bands', pad=20)
plt.ylabel('Count')
save_plot_to_pdf()

# 6. Derived Metrics ======================================================

# Salary per year of experience
exp_mapping = {
    'Entry-level': 1,
    'Mid-level': 4,
    'Senior-level': 8,
    'Executive': 15
}
df['years_experience'] = df['experience_level'].map(exp_mapping)
df['salary_per_year'] = df['salary_in_usd'] / df['years_experience']

# Remote role premium
remote_premium = df.groupby('work_models')['salary_in_usd'].mean().diff().iloc[-1]
metrics_text = f"Remote work premium (average difference): ${remote_premium:,.2f}"

# 7. Top N Lists ==========================================================

market_avg = df['salary_in_usd'].mean()
role_avg = df.groupby('job_title_grouped')['salary_in_usd'].mean()

# Top overpaid/underpaid
overpaid = (role_avg - market_avg).sort_values(ascending=False).head(5)
underpaid = (role_avg - market_avg).sort_values().head(5)

# Remote work adoption
remote_adoption = df[df['work_models'] == 'Remote'].groupby('company_location').size() / df.groupby('company_location').size()

# Experience diversity
exp_diversity = df.groupby('job_title_grouped')['experience_level'].nunique().sort_values(ascending=False)

top_lists_text = f"""
Top 5 Overpaid Roles vs Market Average:
{overpaid.to_string()}

Top 5 Underpaid Roles vs Market Average:
{underpaid.to_string()}

Top 5 Countries with Highest Remote Work Adoption:
{remote_adoption.sort_values(ascending=False).head(5).to_string()}

Top 5 Job Titles with Highest Experience Diversity:
{exp_diversity.head(5).to_string()}
"""
add_text_page(metrics_text + top_lists_text, "Key Metrics and Top Lists")

# Bonus: Reflective Insights ==============================================

highest_paid = df.loc[df['salary_in_usd'].idxmax()]
insights_text = f"""
Global Tech Job Market Insights:
- Average salary across all roles: ${market_avg:,.2f}
- Most common job title: {df['job_title_grouped'].mode()[0]}
- Remote work percentage: {len(df[df['work_models'] == 'Remote'])/len(df)*100:.1f}%

Highest Paid Individual:
- Role: {highest_paid['job_title_grouped']}
- Experience: {highest_paid['experience_level']}
- Location: {highest_paid['company_location']}
- Salary: ${highest_paid['salary_in_usd']:,.2f}

Job Seeker Recommendations:
- Consider remote roles for potential salary premiums
- Target senior-level positions in high-paying countries
- Research company size as it affects compensation
- Some roles have wider salary bands than others
"""
add_text_page(insights_text, "Final Insights and Recommendations")

# Restore stdout and get captured output
sys.stdout = old_stdout
captured_output = new_stdout.getvalue()

# Only add program output page if there's actual content
if captured_output.strip():
    add_text_page(captured_output, "Program Output")

questions_covered = """
Questions Explored in This Analysis:

1. Pattern Discovery:
- How have salaries trended over recent years?
- How has remote work adoption changed over time?
- How does salary growth vary by experience level across different roles?

2. Storytelling with Contrasts:
- How do salaries compare between Data Analysts and Data Scientists?
- What are the work model preferences for different roles?
- How does job title distribution vary by continent?
- How do salaries compare between different regions (e.g., US vs Poland)?

3. Anomaly & Outlier Investigation:
- Which salary values are statistical outliers?
- Where are these outliers concentrated by location and job title?

4. Insight-Driven Questions:
- Which roles offer the highest salaries per experience level?
- How does company size affect salaries?
- Is there a pay premium for remote work?
- Does location matching (company vs employee) affect salaries?

5. Salary Bands Analysis:
- How are different job titles distributed across salary bands?
- How does experience level distribute across salary bands?
- How does company size distribute across salary bands?

6. Derived Metrics:
- What is the salary per year of experience?
- What is the remote work salary premium?

7. Top N Lists:
- Which roles are most overpaid/underpaid vs market average?
- Which countries have highest remote work adoption?
- Which roles have most experience diversity?

Key Recommendations:
1. Consider continental salary differences when job hunting
2. Target roles with highest salary growth potential
3. Remote work offers both flexibility and potential pay premiums
4. Larger companies tend to offer higher salaries for senior roles
5. Some roles have wider salary bands than others - negotiate carefully
"""
add_text_page(questions_covered, "Analysis Questions Covered & Key Takeaways")

# Calculate remote pay differences by role
remote_pay_diff = df.groupby(['job_title_grouped', 'work_models'])['salary_in_usd'].median().unstack()
remote_pay_diff['remote_premium'] = remote_pay_diff['Remote'] - remote_pay_diff['On-site']

# Top 5 roles with biggest remote premiums
top_premium = remote_pay_diff['remote_premium'].sort_values(ascending=False).head(5)
# Top 5 roles with biggest remote penalties
top_penalty = remote_pay_diff['remote_premium'].sort_values().head(5)

# Visualization
plt.figure(figsize=(10, 6))
pd.concat([top_premium, top_penalty]).plot(kind='barh', color=['green']*5 + ['red']*5)
plt.title('Roles with Biggest Remote Pay Premiums (Green) vs Penalties (Red)')
plt.xlabel('Salary Difference (Remote - On-site)')
plt.savefig('remote_pay_gap.png')  # Share this with commenter
plt.show()
# Close the PDF
pdf_pages.close()

from sklearn.preprocessing import MinMaxScaler

# Fix: Create normalized salary column
gdp_data = {'US': 1.0, 'UK': 0.8, 'India': 0.3}  # Example GDP coefficients
df['gdp_factor'] = df['company_location'].map(gdp_data)
df['normalized_salary'] = df['salary_in_usd'] * df['gdp_factor']

# Visualize impact
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='company_location', y='normalized_salary')
plt.title('Salaries Normalized by Country GDP Factor')
plt.savefig('gdp_normalized.png')  





