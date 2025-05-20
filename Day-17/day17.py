#!/usr/bin/env python
# coding: utf-8

# ## Data Science Salaries EDA

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# In[15]:


df = pd.read_csv('data_science_salaries.csv')


# # Basic Dataset Overview

# In[25]:


df.sample(10)


# In[17]:


df.info()


# In[18]:


df = df.drop(["salary_currency","salary"], axis='columns')
df.head()


# In[24]:


cols = ['experience_level','employment_type','work_models','work_year','employee_residence','company_location','company_size','salary_in_usd']

# print(f"{col} - {df[col].unique()}")
output_data = {}
for col in cols:
    output_data[col] = df[col].unique().tolist()
with open("output.json","w") as outfile:
    json.dump(output_data, outfile, indent=4)


# In[28]:


df.shape


# In[29]:


df.info()


# In[ ]:


df.describe() 


# 1. Mean Salary - 145560 
# 2. Maximum Salary - 750000
# 3. Minimum Salary - 15000

# In[31]:


df.columns


# In[32]:


df.isnull().sum()


# In[33]:


df.duplicated().sum()


# In[36]:


with open("employees.json", "w") as outfile:
    json.dump(df['job_title'].value_counts().to_dict(), outfile, indent=4)


# In[37]:


job_title_counts = df['job_title'].value_counts()

print("Unique job titles:", job_title_counts.shape[0])
print("\nTop 10 job titles:\n", job_title_counts.head(10))
print("\nRare job titles (≤ 5 entries):\n", job_title_counts[job_title_counts <= 5])


# In[41]:


plt.figure(figsize=(12,8))
sns.barplot(x=job_title_counts.head(20).values,y=job_title_counts.head(20).index, palette="viridis", legend=False)
plt.title("Top 20 Job Titles")
plt.xlabel("Count")
plt.ylabel("Job Title")
plt.tight_layout()
plt.savefig("JOBs.png")
plt.show()


# In[46]:


df['job_title_grouped'] = df['job_title'].apply(lambda x: x if job_title_counts[x] > 10 else 'Other')
grouped_counts = df['job_title_grouped'].value_counts()

plt.figure(figsize=(13,10))
sns.barplot(x=grouped_counts.values, y=grouped_counts.index)
plt.title("Grouped Job Titles")
plt.xlabel("Count")
plt.ylabel("Job Title Group")
plt.tight_layout()
plt.show()


# In[47]:


plt.figure(figsize=(14, 8))
top_jobs = job_title_counts.head(10).index
sns.boxplot(data=df[df['job_title'].isin(top_jobs)], x='salary_in_usd', y='job_title')
plt.title("Salary Distribution by Top Job Titles")
plt.tight_layout()
plt.show()


# In[48]:


def map_job_title(original_title):
    for group, titles in job_title_mapping.items():
        if original_title in titles:
            return group
    return 'Others'



# In[49]:


df['job_title_grouped'] = df['job_title'].apply(map_job_title)


# In[50]:


plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='job_title_grouped', order=df['job_title_grouped'].value_counts().index, palette='Set2')
plt.title('Grouped Job Titles Frequency')
plt.xlabel('Count')
plt.ylabel('Job Title Group')
plt.tight_layout()
plt.show()


# In[51]:


df['job_title_grouped'].unique()


# In[58]:


plt.figure(figsize=(14,10))
sns.boxplot(data=df, x='job_title_grouped', y='salary_in_usd', palette='viridis')
plt.xticks(rotation=45)
plt.title('Salary Distribution by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Salary (USD)')
plt.tight_layout()
plt.savefig("profilevssalary.jpg")
plt.show()


# In[59]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='experience_level', y='salary_in_usd', palette='Set3')
plt.title('Salary Distribution by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary (USD)')
plt.tight_layout()
plt.savefig("salaryvsexperience")
plt.show()


# In[60]:


top_locations = df['company_location'].value_counts().head(10).index
df_top_locations = df[df['company_location'].isin(top_locations)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top_locations, x='company_location', y='salary_in_usd', palette='coolwarm')
plt.title('Salary Distribution by Company Location (Top 10)')
plt.xlabel('Company Location')
plt.ylabel('Salary (USD)')
plt.tight_layout()
plt.savefig("Locationvssalary")
plt.show()


# In[63]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='work_models', y='salary_in_usd', palette='Pastel1')
plt.title('Salary Distribution by Work Model')
plt.xlabel('Work Model')
plt.ylabel('Salary (USD)')
plt.tight_layout()
plt.savefig("Workmodelvssalary.png")
plt.show()


# In[62]:


df.columns


# In[67]:


plt.figure(figsize=(20,15))
sns.countplot(data=df, y='job_title_grouped', hue='work_models', palette='Set2')
plt.title('Job Roles by Work Model')
plt.xlabel('Count')
plt.ylabel('Job Title')
plt.legend(title='Work Model')
plt.tight_layout()
plt.savefig("workmodelvsjobs.png")
plt.show()


# In[ ]:




