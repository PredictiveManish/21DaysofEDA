# 21 Days of EDA
A 21 Days journey to stronger the ability to analyse and work with data.

---
## 📅 Day 1 – 21-Day EDA Challenge

### 🚀 Goal:
Kickstart the EDA (Exploratory Data Analysis) process by loading the dataset and performing initial inspection using Python and Pandas.
---
## 🧰 Tools Used
- Python
- Pandas
- Jupyter Notebook or any Python IDE

## ✅ Tasks Completed

1. **Environment Setup**
   - Installed `pandas`, `numpy`, `matplotlib`, and `seaborn`
   - Created a new Jupyter Notebook `Day1.ipynb`

2. **Loaded the Dataset**
```python
import pandas as pd

df = pd.read_csv("users.csv")
```
3. Other opeations
```
print(df.shape)       # Rows, Columns
print(df.columns)     # Column names
print(df.dtypes)      # Data types
```
4. Previewing the dataset
```
print(df.isnull().sum())
```
5. Viewed Summary Statistics
```
print(df.describe())
print(df.info())
```
---
