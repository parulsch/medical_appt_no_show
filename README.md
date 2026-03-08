#Appointment No-Show Risk Predictor

## Overview
This project applies the complete data science workflow to a tabular medical appointment dataset to predict whether a patient will miss an appointment (**No-show**). The final deliverable is a deployed Streamlit app that presents descriptive analytics, model performance, explainability, and interactive prediction.

## Problem Statement
Appointment no-shows create operational and financial problems for clinics and other appointment-based businesses. Missed appointments lead to unused time slots, inefficient staffing, longer wait times for other patients, and lost revenue. The goal of this project is to build a predictive tool that helps identify higher-risk appointments so businesses can take actions such as reminders, confirmation calls, and better scheduling decisions.

## Dataset
The project uses the **Medical Appointment No Shows** dataset from Kaggle.  
Each row represents one scheduled appointment and includes patient and appointment information such as age, gender, neighborhood, reminder status, lead time, weekday, and selected health indicators.

### Target Variable
- **No-show**
  - `Yes` = patient missed the appointment
  - `No` = patient attended the appointment

For modeling, the target was converted to:
- `1` = no-show
- `0` = show

## Project Workflow
This project follows the full data science workflow required in HW1:

1. **Descriptive Analytics**
   - dataset introduction
   - target distribution
   - feature distribution and relationship visualizations
   - correlation heatmap

2. **Predictive Analytics**
   - Logistic / Logit baseline
   - Decision Tree with cross-validation
   - Random Forest with cross-validation
   - XGBoost with cross-validation
   - MLP neural network
   - model comparison using Accuracy, Precision, Recall, F1, and AUC

3. **Explainability**
   - SHAP beeswarm plot
   - SHAP bar plot
   - SHAP waterfall plot for a specific prediction

4. **Deployment**
   - Streamlit app with four required tabs
   - models are pre-trained and saved
   - app loads saved artifacts and does not retrain models

## Best Model
The best overall model was **Random Forest**, based on the highest **F1 score**, which is the most appropriate metric for this imbalanced classification problem.  
XGBoost achieved a very similar performance and had the highest AUC.

## Streamlit App Tabs
The deployed Streamlit app includes four tabs:

### Tab 1 — Executive Summary
Provides a non-technical overview of the dataset, business problem, modeling approach, and key findings.

### Tab 2 — Descriptive Analytics
Displays the key visualizations from exploratory data analysis, including target distribution, feature relationships, and the correlation heatmap, along with short interpretation for each plot.

### Tab 3 — Model Performance
Displays the model comparison table, F1 comparison chart, ROC curves, best hyperparameters, and decision tree visualization.

### Tab 4 — Explainability & Interactive Prediction
Displays SHAP plots and allows the user to enter appointment details to get:
- predicted no-show probability
- predicted class
- SHAP waterfall explanation for the custom input

## Repository Structure
```text
medical_appt_no_show/
│
├── app.py
├── train.py
├── train.ipynb
├── requirements.txt
├── data/
│   ├── noshowappointments.csv
├── README.md
├── artifacts.zip
├── artifacts/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── shap/

How to Run Locally

Create and activate a virtual environment, install dependencies, and run the Streamlit app.

Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Requirements

The main dependencies used in this project are:

streamlit

pandas

numpy

scikit-learn==1.6.1

joblib

matplotlib

shap

xgboost

- **GitHub repository link**--- https://github.com/parulsch/medical_appt_no_show
- **Deployed Streamlit app link**--- https://medicalapptnoshow-be5ochuqgsjnacfx3ehyqx.streamlit.app/
