
# **Cancer Model**
### Project Overview

This project is a machine learning model designed to predict whether a given input  or benign based on features extracted from medical data. The model uses various classification algorithms to make accurate predictions, which can help medical professionals in early diagnosis and decision-making processes.

The goal of this project was to implement and evaluate different machine learning models and to understand their strengths and limitations in the context of cancer detection.
## **Technologies Used**

**Programing language:** Python,Html,Css,JavaScript

**Libraries/Frameworks:**
    
*   Flask: lightweight, flexible Python framework for building web applications and APIs with minimal overhead.

*   scikit-learn: For machine learning models and evaluation metrics.

*   pandas: For data manipulation and analysis.

*   matplotlib & seaborn: For data visualization.

*   numpy: For numerical operations.

*   numpy: For numerical operations.

*   joblib & pickle : Python libraries used for serializing and deserializing objects to/from files.

**Dataset:**

*   Breast Cancer Wisconsin Dataset
*   Lung Cancer dataset
*   Colorectal Cancer dataset
*   Prostate Cancer dataset

## **Problem Statement**
Early detection of cancer is crucial for better treatment outcomes. Using a set of diagnostic features from the datasets, the goal is to predict whether a tumor is benign (non-cancerous) or malignant (cancerous).
## **Features**

- **Data Preprocessing:** Handled missing values, normalized the dataset, and performed feature scaling to prepare the data for training.
- **Model Selection:** Implemented multiple classification algorithms such as:
    - Logistic Regression
    - Decision Trees
    - Support Vector Machine (SVM)
    - Random Forest
    - K-Nearest Neighbors (KNN)
- **Evaluation Metrics:** Used accuracy, precision, recall, F1-score, and ROC-AUC for model evaluation.

- Hyperparameter Tuning: Applied GridSearchCV,RandomSearchCV for optimal hyperparameter selection.


##  **How to Run the Project**

1.Clone the repository:

```bash
  git clone https://github.com/yourusername/cancer_model.git
cd cancer_model

```
2.Install dependencies (using pip or conda):

```bash
    pip install -r requirements.txt
```
3.Run the Jupyter notebook (or Python script):

```bash
    jupyter notebook app_notebook.ipynb
```
Or if running the Python script directly:

```bash
    python app.py
```
- Best Model: Random Forest showed the best performance across multiple metrics.



## **Features & Improvements**
- **Model Comparison:** The project allows for easy comparison of different models to choose the best one based on performance metrics.
- **Feature Engineering:** Future improvements can involve more advanced feature engineering, including feature selection and dimensionality reduction techniques like PCA.
- **Deep Learning:** Investigating deep learning models (e.g., CNNs) could be an option for improving accuracy.
## **Files in this Repository**

- **breast_cancer.ipynb:** Jupyter notebook with data loading, preprocessing, model training, and evaluation for the breast cancer.
- **Cancer_model.ipynb:** Jupyter notebook with data loading, preprocessing, model training, and evaluation fro lung Cancer.
- **prostate_cancer.ipynb:** Jupyter notebook with data loading, preprocessing, model training, and evaluation for the prostate cancer.
- **breast-cancer.csv:** dataset for the breast cancer
- **data.csv:** dataset for the prostatecancer
- **survey lung cancer.csv:** dataset for the lung cancer
- **requirements.txt:** List of all required libraries for running the project.
- **app.py:** file is the main entry point for initializing and running a Python web application.
- **static/:** Folder containing the images,javascript and css file in spesific folder.
- **templates/:** containing all html Files
## Learning Outcomes
- **Understanding Machine Learning Algorithms:** Gained hands-on experience with popular machine learning algorithms and their application in a medical context.
- **Data Preprocessing:** Learned techniques for handling and preprocessing real-world datasets, which are often messy and inconsistent.
- **Model Evaluation:** Developed the ability to evaluate machine learning models using various metrics to ensure robustness and reliability.
- **Hyperparameter Tuning:** Applied GridSearchCV to fine-tune models for optimal performance.
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Ananta01Nayk/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ananta-nayak/)

