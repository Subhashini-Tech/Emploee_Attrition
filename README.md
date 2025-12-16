PROJECT: Employee Attrition Analysis and Prediction
Description: This application uses employee data—such as experience, satisfaction levels, promotions, performance, training sessions, and overtime—to predict both attrition risk and performance ratings. By highlighting the key factors that influence why employees leave or how they perform, the system provides HR with actionable insights to reduce turnover, improve employee engagement, and strengthen workforce planning.
Technologies Used: Python, Streamlit, ML, EDA
Project Details:
Data Preparation & Pre-processing
•	Analyzed and cleaned the employee attrition dataset.
•	Removed irrelevant and redundant features to optimize model performance.
•	Applied:
o	Label Encoding for categorical variables
o	Log Transformation to handle skewness and outliers
•	Engineered new features such as Total Satisfaction Score for better predictive power.
EDA
•	Performed in-depth EDA using Pandas to identify patterns and trends.
•	Identified key drivers influencing employee attrition.
Visualization
•	Built an interactive Streamlit dashboard for HR stakeholders.
•	Visualized:
o	Attrition distribution by satisfaction and job involvement
o	Heatmaps showing attrition rate vs performance and satisfaction
o	KPI metrics such as total employees and attrition rate
Machine Learning Models
•	Trained and evaluated multiple models for Attrition Prediction and Performance Prediction:
o	Logistic Regression
o	Random Forest
o	Decision Tree
o	K-Nearest Neighbors (KNN)
o	Support Vector Machine (SVM)
o	AdaBoost
o	Gradient Boosting
o	XGBoost
Model Evaluation Metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	AUC-ROC
Model Selection & Deployment
•	Selected the best-performing model based on evaluation metrics.
•	Saved the trained model using Joblib/Pickle.
•	Integrated the model into Streamlit for:
o	Attrition prediction
o	Performance rating prediction
