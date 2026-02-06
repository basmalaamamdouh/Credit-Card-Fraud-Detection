# Credit Card Fraud Detection: Machine Learning & Deep Learning Study 

## 📌 Project Overview
This project focuses on identifying fraudulent credit card transactions using a highly imbalanced dataset. As the **Team Leader**, I led the implementation of various advanced techniques to handle data imbalance, ensuring that our models can accurately detect fraud while minimizing false alarms.

## 📊 Dataset Information
The dataset used in this project is the **Credit Card Fraud Detection** dataset provided by **Worldline and the ULB**.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Statistics:** 284,807 transactions, only **492** are frauds (**0.172%**).
* **Features:** Contains 31 numerical features. Features **V1-V28** are principal components obtained via **PCA** for confidentiality. **'Time'**, **'Amount'**, and the target **'Class'** (1 for fraud, 0 for normal) are the only non-transformed features.

[<img width="1077" height="546" alt="image" src="https://github.com/user-attachments/assets/a65c0b0a-2492-4a41-8909-a3bac75cc166" />]

## 🛠️ Tech Stack
* **Language:** Python
* **Frameworks:** TensorFlow, Keras
* **Libraries:** Scikit-Learn, Imbalanced-Learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 📉 Methodology & Pipeline

### 1. Data Cleaning & Outlier Removal
* Used **Heatmaps** to identify correlations between features and the target class.
* Applied the **Interquartile Range (IQR)** method to remove extreme outliers from highly correlated features (V14, V12, V10), which significantly improved model stability.

### 2. Dimensionality Reduction & Clustering
* Implemented **t-SNE, PCA, and Truncated SVD** to visualize the clusters.
* **t-SNE** successfully separated fraud and non-fraud cases in a 2D space, indicating that our predictive models would perform well.

[<img width="1292" height="400" alt="image" src="https://github.com/user-attachments/assets/a555c36f-aeeb-4da2-a46c-dac6e12e9c86" />]

### 3. Handling Class Imbalance
* **Random Under-sampling:** Created a 50/50 balanced sub-sample to analyze feature relations without bias.
* **SMOTE (Over-sampling):** Synthetically generated new fraud cases to train models on the full dataset, preserving all information from the majority class.
* **NearMiss Technique:** Used intelligent under-sampling to focus the model on difficult classification boundaries.

### 4. Machine Learning & Deep Learning Modeling
* **Classifiers:** Trained and optimized **Logistic Regression, SVC, KNN, and Decision Trees** using **GridSearchCV** for hyperparameter tuning.
* **Deep Learning:** Developed an **Artificial Neural Network (ANN)** using Keras, trained on both undersampled and oversampled data, utilizing **Tesla T4 GPUs** for accelerated training.

## 📊 Key Results
| Model | Technique | Recall (Fraud Detection) | F1-Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Under-sampling | ~94% | Medium |
| **Neural Network** | **SMOTE** | **High** | **High** |

*The Neural Network trained with SMOTE provided the most robust results, achieving a high balance between catching fraud (Recall) and maintaining precision.*

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Credit-Card-Fraud-Detection.git](https://github.com/YourUsername/Credit-Card-Fraud-Detection.git)
