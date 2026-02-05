# 🫀 Heart Disease Prediction Project

## 📌 Overview
This project focuses on **predicting heart disease** using machine learning models.  
We perform **Exploratory Data Analysis (EDA)**, **data cleaning**, **feature engineering**, and train multiple classification models to identify patients at risk of heart disease.  

The dataset used is `heart.csv` with **918 rows and 12 columns** containing both categorical and numerical features.

---

## 📊 Dataset Information
- **Rows:** 918  
- **Columns:** 12  
- **Target Variable:** `HeartDisease` (0 = No, 1 = Yes)  

### Features:
- Age  
- Sex  
- ChestPainType  
- RestingBP  
- Cholesterol  
- FastingBS  
- RestingECG  
- MaxHR  
- ExerciseAngina  
- Oldpeak  
- ST_Slope  
- HeartDisease  

---

## 🧹 Data Preprocessing
1. **Handling Errors in Data**
   - Replaced `0` values in **Cholesterol** and **RestingBP** with their respective mean values.
2. **Encoding Categorical Variables**
   - Used `pd.get_dummies()` with `drop_first=True`.
3. **Scaling Numerical Features**
   - Applied `StandardScaler` to normalize numerical columns.

---

## 📈 Exploratory Data Analysis (EDA)
- Histograms for Age, RestingBP, Cholesterol, MaxHR  
- Countplots for categorical variables (Sex, ChestPainType, FastingBS)  
- Boxplots & Violin plots for distribution analysis  
- Heatmap for correlation among numerical features  

---

## 🤖 Machine Learning Models
We trained and evaluated the following models:
- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Support Vector Machine (SVM)  

### 📊 Model Performance
| Model               | Accuracy | F1-Score |
|----------------------|----------|----------|
| Logistic Regression  | 0.8967   | 0.9124   |
| Decision Tree        | 0.8098   | 0.8293   |
| KNN                  | 0.8641   | 0.8837   |
| Naive Bayes          | 0.8804   | 0.8962   |
| **SVM (Best Model)** | **0.9076** | **0.9238** |

---

## 💾 Model Saving
The best performing model (**SVM**) was saved using `joblib`:
- `heart_disease_model.pkl` → Trained SVM model  
- `scaler.pkl` → StandardScaler object  
- `columns.pkl` → Feature column names  

---

## 🚀 How to Run
### 1. Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook / script
```bash
python heart_disease_analysis.py
```

### 4. Load the saved model
```python
import joblib

# Load model, scaler, and columns
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Example prediction
import numpy as np
sample = np.array([[54, 130, 223, 0, 138, 0.6, 1, 0, 1, 0, 1, 0, 0, 0, 1]])  # Example input
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("Heart Disease Prediction:", prediction)
```

---

## 📦 Requirements
- numpy  
- pandas  
- seaborn  
- matplotlib  
- scikit-learn  
- joblib  
- sheryanalysis (optional for quick EDA)  

Install all dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn joblib sheryanalysis
```

---

## 👨‍💻 Author

**Ayushmaan-99**

## 📌 Future Improvements
- Hyperparameter tuning for better accuracy  
- Cross-validation for robust evaluation  
- Deployment as a **Flask/Django web app** or **Streamlit dashboard**  

---

## 🏷️ License
This project is licensed under the MIT License.  

---
