# AI-Powered-Attrition-Prediction-System

A complete mini-project that uses machine learning to predict employee attrition (whether an employee will leave the company), built as part of an AI internship technical task.

## 🎯 Objective

- Clean and preprocess real-world HR data
- Build and evaluate classification models
- Visualize key insights and metrics
- Explore a trending AI topic (RAG)

---

## 📂 Dataset

**IBM HR Analytics Employee Attrition Dataset**  
Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 📊 Project Structure

### 1️⃣ Data Preprocessing

🔧 Workflow:
Loaded and explored the IBM HR Analytics dataset

Handled missing values (none)

Dropped redundant columns (e.g., Over18, EmployeeCount)

Categorical encoding using LabelEncoder and OneHotEncoder

Scaled numerical features with StandardScaler

Performed outlier detection and feature cleanup

Visualized:

Age vs Attrition

Department-wise Attrition

Monthly Income by Job Role

🎯 Results:
Best model: Random Forest Classifier

Train/Test split: 80:20

Accuracy: ~87.2%

Evaluation:

Confusion Matrix

Precision, Recall, F1-Score

ROC Curve (AUC)

### 2 Model Building and Evaluation
Objective: Build and evaluate a robust classification model to detect attrition.

🔧 Workflow:
Built a Random Forest Classifier with tuned parameters

Compared results with Logistic Regression

Hyperparameters: n_estimators=100, max_depth=10, random_state=42

Evaluated using:

Accuracy, Precision, Recall, F1-Score

Confusion Matrix (visualized using Seaborn)

ROC Curve (via scikit-learn's metrics)

📊 Results:
Metric	Value
Accuracy	87.2%
Precision	86.8%
Recall	84.7%
F1 Score	85.7%


### 3️⃣ Research Brief (RAG - Retrieval-Augmented Generation)

PDF Summary included:  
- 🔍 What is RAG?  
- 💡 Why it matters?  
- 🧪 Real-world Use Case  
- ⚠️ Challenges

📎 File: `RAG_AI_Research_Brief.pdf`

---

## 📁 Files Included

| File                              | Description                                |
|-----------------------------------|--------------------------------------------|
| `AI_Attrition_Predictor.ipynb`    | Jupyter notebook with full code & results  |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | Original dataset                  |
| `RAG_AI_Research_Brief.pdf`       | Research brief on AI topic (RAG)           |
| `README.md`                       | Project summary and documentation          |

---

## 💻 Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Jupyter Notebook
- Machine Learning (Random Forest Classifier)
- AI Research Topic: Retrieval-Augmented Generation (RAG)

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/kesavanelumalai/AI-Powered-Attrition-Prediction-System
