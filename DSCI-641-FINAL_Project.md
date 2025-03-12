```python
pip install tqdm

```

    Requirement already satisfied: tqdm in /opt/anaconda3/lib/python3.11/site-packages (4.65.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

```

Load data


```python
# Load dataset
data = pd.read_csv('./Data/medical_data.csv')
```


```python
# dataset overview
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>DateOfBirth</th>
      <th>Gender</th>
      <th>Symptoms</th>
      <th>Causes</th>
      <th>Disease</th>
      <th>Medicine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>241</td>
      <td>241</td>
      <td>242</td>
      <td>247</td>
      <td>245</td>
      <td>249</td>
      <td>242</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>87</td>
      <td>98</td>
      <td>4</td>
      <td>53</td>
      <td>62</td>
      <td>68</td>
      <td>65</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Sophia Koh</td>
      <td>05-10-1999</td>
      <td>Male</td>
      <td>Fatigue, Weakness</td>
      <td>Food Poisoning</td>
      <td>Gastroenteritis</td>
      <td>Rest, Lifestyle</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>9</td>
      <td>8</td>
      <td>116</td>
      <td>19</td>
      <td>20</td>
      <td>20</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 287 entries, 0 to 286
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   Name         241 non-null    object
     1   DateOfBirth  241 non-null    object
     2   Gender       242 non-null    object
     3   Symptoms     247 non-null    object
     4   Causes       245 non-null    object
     5   Disease      249 non-null    object
     6   Medicine     242 non-null    object
    dtypes: object(7)
    memory usage: 15.8+ KB
    None



```python
# Display the first few rows of the dataset
print("Initial Data Overview:")
print(data.head())
```

    Initial Data Overview:
              Name DateOfBirth  Gender             Symptoms               Causes  \
    0     John Doe  15-05-1980    Male         Fever, Cough      Viral Infection   
    1   Jane Smith  10-08-1992  Female    Headache, Fatigue               Stress   
    2  Michael Lee  20-02-1975    Male  Shortness of breath            Pollution   
    3   Emily Chen  03-11-1988  Female     Nausea, Vomiting       Food Poisoning   
    4    Alex Wong  12-06-2001    Male          Sore Throat  Bacterial Infection   
    
               Disease           Medicine  
    0      Common Cold    Ibuprofen, Rest  
    1         Migraine        Sumatriptan  
    2           Asthma  Albuterol Inhaler  
    3  Gastroenteritis   Oral Rehydration  
    4     Strep Throat         Penicillin  



```python
# Drop non-numeric columns
data = data.drop(columns=['Name', 'DateOfBirth'])
```


```python
# Handle categorical columns
label_encoders = {}
for column in ['Gender', 'Symptoms', 'Causes', 'Disease', 'Medicine']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le
```


```python
# Check for missing values
print("\nMissing Values Check:")
print(data.isnull().sum())
```

    
    Missing Values Check:
    Gender      0
    Symptoms    0
    Causes      0
    Disease     0
    Medicine    0
    dtype: int64



```python
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')  
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
```


```python
# Check if there are any missing values remaining
print("\nMissing Values Check After Imputation:")
print(data.isnull().sum())
```

    
    Missing Values Check After Imputation:
    Gender      0
    Symptoms    0
    Causes      0
    Disease     0
    Medicine    0
    dtype: int64


EDA


```python
# Distribuution of symptoms
medical_data_symptoms = pd.DataFrame({
    'Symptoms': ['Fever', 'Headache', 'Shortness of breath', 'Nausea', 'Sore Throat', 
                 'Fever', 'Headache', 'Fever', 'Nausea', 'Sore Throat']
})

# individual symptoms to broader categories
symptom_mapping = {
    'Fever': 'Fever',
    'Headache': 'Headache',
    'Shortness of breath': 'Respiratory Issues',
    'Nausea': 'Digestive Issues',
    'Sore Throat': 'Throat Issues'
}

# individual symptoms to broader categories
medical_data_symptoms['Symptoms'] = medical_data_symptoms['Symptoms'].map(symptom_mapping)

# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=medical_data_symptoms, y='Symptoms', palette='viridis')
plt.title('Distribution of Symptoms')
plt.xlabel('Count')
plt.ylabel('Symptoms')
plt.show()
```


    
![png](output_13_0.png)
    



```python
# causes distribution
medical_data_causes = pd.DataFrame({
    'Causes': ['Viral Infection', 'Stress', 'Pollution', 'Food Poisoning', 'Bacterial Infection',
               'Viral Infection', 'Stress', 'Food Poisoning', 'Viral Infection', 'Bacterial Infection']
})

#causes to broader categories
cause_mapping = {
    'Viral Infection': 'Viral Infections',
    'Stress': 'Stress',
    'Pollution': 'Environmental Issues',
    'Food Poisoning': 'Digestive Issues',
    'Bacterial Infection': 'Bacterial Infections'
}

# Map causes 
medical_data_causes['Causes'] = medical_data_causes['Causes'].map(cause_mapping)

# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=medical_data_causes, y='Causes', palette='viridis')
plt.title('Distribution of Causes')
plt.xlabel('Count')
plt.ylabel('Causes')
plt.show()
```


    
![png](output_14_0.png)
    



```python
# Disease
medical_data_diseases = pd.DataFrame({
    'Disease': ['Common Cold', 'Migraine', 'Asthma', 'Gastroenteritis', 'Strep Throat',
                'Common Cold', 'Migraine', 'Asthma', 'Common Cold', 'Strep Throat']
})

# diseases to broader categories
disease_mapping = {
    'Common Cold': 'Respiratory Infections',
    'Migraine': 'Headaches',
    'Asthma': 'Respiratory Conditions',
    'Gastroenteritis': 'Digestive Issues',
    'Strep Throat': 'Throat Infections'
}

# Map of diseases 
medical_data_diseases['Disease'] = medical_data_diseases['Disease'].map(disease_mapping)

# Create the count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=medical_data_diseases, y='Disease', palette='viridis')
plt.title('Distribution of Diseases')
plt.xlabel('Count')
plt.ylabel('Disease')
plt.show()
```


    
![png](output_15_0.png)
    



```python

```


```python
# Separate features and target variable
X = data.drop(columns='Medicine')
y = data['Medicine']
```


```python
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=42)</pre></div></div></div></div></div>




```python
# Predict and evaluate
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
```


```python
# Print classification report with zero_division handling
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt, zero_division=1))
```

    
    Decision Tree Classification Report:
                  precision    recall  f1-score   support
    
               2       0.00      1.00      0.00         0
               5       0.00      1.00      0.00         0
               9       1.00      0.00      0.00         1
              10       1.00      0.00      0.00         1
              13       1.00      0.00      0.00         1
              15       0.00      1.00      0.00         0
              19       1.00      1.00      1.00         1
              20       1.00      0.50      0.67         2
              22       1.00      1.00      1.00         1
              23       1.00      1.00      1.00         3
              27       1.00      1.00      1.00         2
              28       1.00      1.00      1.00         1
              29       0.50      1.00      0.67         1
              31       0.60      1.00      0.75         3
              32       1.00      1.00      1.00         4
              35       0.00      1.00      0.00         0
              38       1.00      1.00      1.00         2
              40       1.00      0.00      0.00         1
              41       1.00      1.00      1.00         3
              44       1.00      0.67      0.80         3
              45       1.00      1.00      1.00         2
              46       0.50      0.50      0.50         2
              47       1.00      1.00      1.00         2
              50       1.00      1.00      1.00         2
              51       0.83      1.00      0.91         5
              53       1.00      1.00      1.00         1
              54       1.00      0.00      0.00         1
              57       1.00      0.67      0.80         3
              62       1.00      1.00      1.00        10
    
        accuracy                           0.84        58
       macro avg       0.81      0.77      0.62        58
    weighted avg       0.94      0.84      0.84        58
    



```python
# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

```




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>




```python
# Predict and evaluate
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
```


```python
print(f"\nRandom Forest Accuracy: {accuracy_rf}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=1))

```

    
    Random Forest Accuracy: 0.8793103448275862
    
    Random Forest Classification Report:
                  precision    recall  f1-score   support
    
               5       0.00      1.00      0.00         0
               9       1.00      0.00      0.00         1
              10       1.00      0.00      0.00         1
              13       1.00      0.00      0.00         1
              15       0.00      1.00      0.00         0
              19       1.00      1.00      1.00         1
              20       1.00      1.00      1.00         2
              21       0.00      1.00      0.00         0
              22       1.00      1.00      1.00         1
              23       1.00      1.00      1.00         3
              27       1.00      1.00      1.00         2
              28       1.00      1.00      1.00         1
              29       0.50      1.00      0.67         1
              31       0.75      1.00      0.86         3
              32       1.00      1.00      1.00         4
              35       0.00      1.00      0.00         0
              38       1.00      1.00      1.00         2
              40       1.00      1.00      1.00         1
              41       1.00      1.00      1.00         3
              44       1.00      0.67      0.80         3
              45       1.00      1.00      1.00         2
              46       1.00      0.50      0.67         2
              47       1.00      1.00      1.00         2
              50       1.00      1.00      1.00         2
              51       0.83      1.00      0.91         5
              53       1.00      1.00      1.00         1
              54       1.00      0.00      0.00         1
              57       1.00      0.67      0.80         3
              62       1.00      1.00      1.00        10
    
        accuracy                           0.88        58
       macro avg       0.83      0.82      0.68        58
    weighted avg       0.96      0.88      0.88        58
    



```python
# Initialize and train the Logistic Regression model
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train, y_train)

```

    /opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=1000, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=1000, random_state=42)</pre></div></div></div></div></div>




```python
# Predict and evaluate
y_pred_lr = lr_classifier.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
```


```python
print(f"\nLogistic Regression Accuracy: {accuracy_lr}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=1))
```

    
    Logistic Regression Accuracy: 0.8275862068965517
    
    Logistic Regression Classification Report:
                  precision    recall  f1-score   support
    
               9       1.00      0.00      0.00         1
              10       1.00      0.00      0.00         1
              13       1.00      0.00      0.00         1
              19       1.00      1.00      1.00         1
              20       1.00      0.50      0.67         2
              22       1.00      1.00      1.00         1
              23       1.00      1.00      1.00         3
              27       1.00      1.00      1.00         2
              28       1.00      1.00      1.00         1
              29       0.50      1.00      0.67         1
              31       0.75      1.00      0.86         3
              32       1.00      1.00      1.00         4
              35       0.00      1.00      0.00         0
              38       0.67      1.00      0.80         2
              40       1.00      1.00      1.00         1
              41       0.50      1.00      0.67         3
              44       1.00      0.67      0.80         3
              45       1.00      0.00      0.00         2
              46       0.50      0.50      0.50         2
              47       1.00      1.00      1.00         2
              50       1.00      1.00      1.00         2
              51       0.83      1.00      0.91         5
              53       1.00      1.00      1.00         1
              54       1.00      0.00      0.00         1
              57       1.00      0.67      0.80         3
              62       1.00      1.00      1.00        10
    
        accuracy                           0.83        58
       macro avg       0.88      0.74      0.68        58
    weighted avg       0.91      0.83      0.80        58
    



```python
# Plotting model performance
models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
accuracies = [accuracy_dt, accuracy_rf, accuracy_lr]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
```

    /opt/anaconda3/lib/python3.11/site-packages/seaborn/_oldcore.py:1765: FutureWarning: unique with argument that is not not a Series, Index, ExtensionArray, or np.ndarray is deprecated and will raise in a future version.
      order = pd.unique(vector)



    
![png](output_28_1.png)
    



```python
#  Random Forest
rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
print(f"Random Forest Cross-Validation Scores: {rf_cv_scores}")
print(f"Mean Cross-Validation Score: {rf_cv_scores.mean()}")

```

    /opt/anaconda3/lib/python3.11/site-packages/sklearn/model_selection/_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(


    Random Forest Cross-Validation Scores: [0.82758621 0.86206897 0.80701754 0.87719298 0.85964912]
    Mean Cross-Validation Score: 0.8467029643073201


Model Tuning and Optimization
Hyperparameter Tuning
Optimize your model’s performance by tuning hyperparameters.
Use techniques like Grid Search or Random Search to find the best parameters.


```python
#Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

```

    /opt/anaconda3/lib/python3.11/site-packages/sklearn/model_selection/_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(


    Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}


Evaluate feature importance and consider adding or removing features to improve model performance.
Explore techniques like feature scaling, encoding categorical variables, and generating new features.


```python
importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance (Random Forest):")
print(feature_importance_df)
```

    
    Feature Importance (Random Forest):
        Feature  Importance
    3   Disease    0.312419
    1  Symptoms    0.306058
    2    Causes    0.269709
    0    Gender    0.111815


Model Validation
Check for Overfitting/Underfitting-
Evaluate learning curves to check if our model is overfitting or underfitting.
Consider adjusting model complexity based on these insights.


```python

train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X, y, cv=5, scoring='accuracy', n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid()
plt.show()

```

    /opt/anaconda3/lib/python3.11/site-packages/sklearn/model_selection/_split.py:700: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(



    
![png](output_35_1.png)
    



```python

```


```python

```
