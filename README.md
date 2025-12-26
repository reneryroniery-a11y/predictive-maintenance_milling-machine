# 🏭 Industrial Predictive Maintenance with GLM & Physics-Informed Features

This project applies advanced statistical techniques and machine learning to predict failures of a milling machine, aiming to minimize downtime due to corrective maintenance.

## 💼 Business Problem
Unplanned failures usually costs a lot of money for industries. This project's objective is to develop a classifier model that can predict a failure state of a milling machine before the failure occurs.
* **Challenge:** Severe unbalanced data with only 3% of failures
* **Goal:** Maximize the 'recall' (to capture as much failures as possible) while maintaining an acceptable operational precision.

## 🛠️ Tools and Methodologies
This project follows a robust pipeline of data engineering and statistics:

* **Languages and Libraries:** Python (Pandas, Numpy, Statsmodels, Scikit-learn).
* **Model:** Generalized Linear Models (GLM) - Binary Logistic Regression.
* **Feature Engineering:** Creation of physical variables (Power and Temperature Gradient).
* **Unbalance Treatment:** By applying **SMOTE** technique (Synthetic Minority Over-sampling Technique).
* **Statistical Inference:** Variables selection through p-value analysis and odds-ratio interpretation of the coeficients

## 📊 Key Results

The final model surpassed the baseline, providing statistical rigor and operational value:

| Metric | Performance | Interpretation |
| :--- | :--- | :--- |
| **Recall (Classe Falha)** | **81%** | 4 out of 5 potential failures are detected. |
| **AUC-ROC** | **0.91** | High capability of differentiating norma operation from failure state|
| **False Negatives** | Low | Great reduction of unplanned failures risk. |

### Engineering Insights
The coeficient analysis confirmed the physical hypothesis:
1. **Torque** is the most critical factor for failure prediction (+8.44 log-odds).
2. **Temperature Gradient** is more statistically relevant than the absolute temperature.
   
## 🚀 How to run the code:
1. Copy the repository.
2. Install the libraries: `pip install pandas numpy statsmodels imbalanced-learn seaborn`
3. Run the notebook `milling_machine_failure_prediction_GLM.ipynb`.

---
*Author: Renery Roniery de Souza Carvalho - Mechanical Engineer and Data Scientist
