# Tourism Package Prediction (MLOps Project)

## Problem Statement
The objective of this project is to predict whether a customer will purchase a tourism package based on their demographic and behavioral data.

---

##  Dataset
The dataset contains customer-related features such as:

- Age  
- Monthly Income  
- Number of Trips  
- Passport Availability  
- City Tier  

Target Variable:
- **ProdTaken** (1 = Purchased, 0 = Not Purchased)

---

## Approach

### 1. Data Preprocessing
- Selected relevant features
- Removed unnecessary columns
- Handled class imbalance using SMOTE

### 2. Model Building
- Algorithm used: Random Forest Classifier (Scikit-learn)  
- Trained on balanced dataset  
- Saved model as `model.pkl`

### 3. Model Evaluation
- Used classification metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  

---

## MLOps Pipeline

This project follows a complete MLOps workflow:

### Training Automation
- Implemented CI/CD using GitHub Actions  
- Model retrains automatically on every push to `main`

### Model Storage
- Model is uploaded and versioned on Hugging Face Hub  

### Deployment
- Built an interactive UI using Streamlit  
- Deployed on Hugging Face Spaces  
- Model is dynamically downloaded during runtime  

---

## Application Features

- User inputs:
  - Age  
  - Monthly Income  
  - Number of Trips  
  - Passport  
  - City Tier  

- Outputs:
  - Purchase Prediction  
  - Purchase Probability  

---

## Workflow

1. Data ingestion  
2. Data preprocessing  
3. Handling imbalance (SMOTE)  
4. Model training  
5. Model evaluation  
6. Model saving (`model.pkl`)  
7. CI/CD pipeline execution  
8. Model upload to Hugging Face  
9. Streamlit app deployment  

---

## Live Application

https://huggingface.co/spaces/Arjuna3667/tourism-prediction-app

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn  
- Streamlit  
- GitHub Actions  
- Hugging Face  

---

## Key Highlights

- End-to-end ML pipeline implementation  
- Automated training and deployment  
- Real-time prediction system  
- Production-style project setup  

---

## Conclusion

This project demonstrates how to build, automate, and deploy a machine learning model using modern MLOps practices. It provides a scalable and reproducible solution for predicting customer purchase behavior.
