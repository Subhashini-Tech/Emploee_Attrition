import streamlit as st
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
import pickle, os, tempfile 
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")


side = st.sidebar.radio(
    "Navigation",
    ["HOME", "SUMMARY", "INSIGHTS - HR", "ATTRITION PREDICTION","RATINGS PREDICTION"]
)

# Read data
data = pd.read_csv(r"C:\Users\Admin\Employee-Attrition.csv")

# Dropped the unused featurers
data.drop(['DailyRate','EmployeeCount','HourlyRate','MonthlyRate','Over18','StandardHours','EmployeeNumber'],axis = 1, inplace = True)

# Splitted the features based on data into 3 data sets
data_scale = data[['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked','PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
data_class = data[['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']]
data_rate = data[['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']]

# Feature Engineering
data_rate['Total_Satisfaction'] = (data_rate['EnvironmentSatisfaction'] + 
                            data_rate['JobInvolvement'] + 
                            data_rate['JobSatisfaction'] + 
                            data_rate['RelationshipSatisfaction'] +
                            data_rate['WorkLifeBalance']) /5 
data_rate.drop(['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'], axis=1, inplace=True)

# Label Encoded the data set having non linear values
for col in data_class.columns:
    le = LabelEncoder()
    data_class[col] = le.fit_transform(data_class[col])
#data_class

# To find Outliers in data set having linear values
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15,15))
axes = axes.flatten()

for ax, col in zip(axes,data_scale.columns):
    sns.boxplot(data_scale[col], ax=ax)

for ax in axes[len(data_scale.columns):]:
    ax.set_visible(False)

plt.show()
plt.tight_layout()

# To fetch only the outlier columns in data_scale
outlier_cols = []

for col in data_scale.columns:
    Q1 = data_scale[col].quantile(0.25)
    Q3 = data_scale[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # check if any values are outside bounds
    if ((data_scale[col] < lower) | (data_scale[col] > upper)).any():
        outlier_cols.append(col)

#outlier_cols

# Used Log Transformation to handle outliers
for col in outlier_cols:
    # Shift column if needed
    if (data_scale[col] <= 0).any():
        data_scale[col] = data_scale[col] - data_scale[col].min() + 1
        
    data_scale[col] = np.log10(data_scale[col])
   
#data_scale

# After transformation data_scale
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15,15))
axes = axes.flatten()

for ax, col in zip(axes,data_scale.columns):
    sns.boxplot(data_scale[col], ax=ax) 

plt.tight_layout()
plt.show()

# To find the outlier in data_rate
outlier_cols1 = []

for col in data_rate.columns:
    Q1 = data_rate[col].quantile(0.25)
    Q3 = data_rate[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # check if any values are outside bounds
    if ((data_rate[col] < lower) | (data_rate[col] > upper)).any():
        outlier_cols1.append(col)

#outlier_cols1

# Used Log Transformation to handle outliers in data_rate
for col in outlier_cols1:
    # Shift column if needed
    if (data_rate[col] <= 0).any():
        data_rate[col] = data_rate[col] - data_rate[col].min() + 1
        
    data_rate[col] = np.log10(data_rate[col])
   
#data_rate

# After transformation data_rate
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
axes = axes.flatten()

for ax, col in zip(axes,data_rate.columns):
    sns.boxplot(data_rate[col], ax=ax) 

plt.tight_layout()
plt.show()

# Concatenate the data sets with transformed data --> new cleaned data set data_attrition
data_attrition = pd.concat([data_class, data_rate, data_scale], axis=1)
#data_attrition

# Logistic Regression ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# Create a pipeline:
pipeline = Pipeline(steps=[
    #('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy Precision Recall F1 AUC-ROC - Logistic Regression
print(classification_report(y_test, y_pred))
accuracy_LR = accuracy_score(y_test, y_pred)
precision_LR = precision_score(y_test, y_pred, average='weighted')
recall_LR = recall_score(y_test, y_pred, average='weighted')
f1_LR = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_LR = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_LR)
print("Precision:", precision_LR)
print("Recall:", recall_LR)
print("F1 Score:", f1_LR)
print("AUC-ROC Score:", auc_roc_LR)

# Random Forest ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
# Create a pipeline:
pipeline = Pipeline(steps=[
    #('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    #('scaler', StandardScaler()),
    ('model', RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        random_state=42
    ))
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy Precision Recall F1 AUC-ROC - Random Forest
print(classification_report(y_test, y_pred))
accuracy_RF = accuracy_score(y_test, y_pred)
precision_RF = precision_score(y_test, y_pred, average='weighted')
recall_RF = recall_score(y_test, y_pred, average='weighted')
f1_RF = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_RF = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_RF)
print("Precision:", precision_RF)
print("Recall:", recall_RF)
print("F1 Score:", f1_RF)
print("AUC-ROC Score:", auc_roc_RF)

# Decision Tree ML
# Declare target and featres
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=123)
# Create a pipeline:
pipeline = Pipeline(steps=[
    #('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    #('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier(
        criterion='entropy',     # or "entropy" "gini" "log_loss"
        max_depth=5,
        random_state=123
    ))
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

#Accuracy Precision Recall F1 AUC-ROC - Decision Tree
print(classification_report(y_test, y_pred))
accuracy_DT = accuracy_score(y_test, y_pred)
precision_DT = precision_score(y_test, y_pred, average='weighted')
recall_DT = recall_score(y_test, y_pred, average='weighted')
f1_DT = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_DT = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_DT)
print("Precision:", precision_DT)
print("Recall:", recall_DT)
print("F1 Score:", f1_DT)
print("AUC-ROC Score:", auc_roc_DT)

# KNN ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=123)
# Create a pipeline:
pipeline = Pipeline(steps=[
    #('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(
        n_neighbors=12,     # default K
        weights='distance', # or 'distance' 'uniform'
        metric='manhattan' # or 'euclidean', 'manhattan' 'minkowski'
    ))
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy Precision Recall F1 AUC-ROC - KNN
print(classification_report(y_test, y_pred))
accuracy_KNN = accuracy_score(y_test, y_pred)
precision_KNN = precision_score(y_test, y_pred, average='weighted')
recall_KNN = recall_score(y_test, y_pred, average='weighted')
f1_KNN = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_KNN = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_KNN)
print("Precision:", precision_KNN)
print("Recall:", recall_KNN)
print("F1 Score:", f1_KNN)
print("AUC-ROC Score:", auc_roc_KNN)

# Support Vector Machine ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=123)
# Create a pipeline:
pipeline = Pipeline(steps=[
    #('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    ('scaler', StandardScaler()),
    ('model', SVC(
        kernel='rbf',        # common choice
        C=0.01,               # regularization
        gamma=0.01,       # 'scale' 'auto' 0.01
        probability=True,    # needed if you want ROC curve
        random_state=99
    ))
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy Precision Recall F1 AUC-ROC - Support Vector Machine
print(classification_report(y_test, y_pred))
accuracy_SVM = accuracy_score(y_test, y_pred)
precision_SVM = precision_score(y_test, y_pred, average='weighted')
recall_SVM = recall_score(y_test, y_pred, average='weighted')
f1_SVM = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_SVM = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_SVM)
print("Precision:", precision_SVM)
print("Recall:", recall_SVM)
print("F1 Score:", f1_SVM)
print("AUC-ROC Score:", auc_roc_SVM)

# Ada Boosting ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=99)
# Create a pipeline:
pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    ('scaler', StandardScaler()),
    ('model', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            criterion='gini', # 'gini' 'entropy' 'log_loss'
            max_depth=2,    # very common for AdaBoost
            random_state=123
        ),
        n_estimators=350,   # number of boosting rounds
        learning_rate=0.1,  # shrinkage parameter
        random_state=99
    ))
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy Precision Recall F1 AUC-ROC - Ada Boosting
print(classification_report(y_test, y_pred))
accuracy_AB = accuracy_score(y_test, y_pred)
precision_AB = precision_score(y_test, y_pred, average='weighted')
recall_AB = recall_score(y_test, y_pred, average='weighted')
f1_AB = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_AB = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_AB)
print("Precision:", precision_AB)
print("Recall:", recall_AB)
print("F1 Score:", f1_AB)
print("AUC-ROC Score:", auc_roc_AB)

# Gradient Boosting ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=123)
# Create a pipeline:
pipeline_GB = Pipeline(steps=[
    #('smote', SMOTE(sampling_strategy="not majority", random_state=42)),
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=300,      # number of boosting trees
        learning_rate=0.01,     # shrinkage / step size
        max_depth=5,           # depth of individual trees
        subsample=0.5,         # 1.0 = no stochasticity
        random_state=99
    ))
])

# Fit on training data
pipeline_GB.fit(X_train, y_train)

# Predict
y_pred = pipeline_GB.predict(X_test)

# Save as pickle
#with open("gradient_boosting_model.pkl", "wb") as f:
    #pickle.dump(pipeline_GB, f)

# with tempfile.NamedTemporaryFile(delete=False) as tmp:
#     pickle.dump(pipeline_GB, tmp)
#     temp_name = tmp.name

# os.replace(temp_name, "gradient_boosting_model.pkl")   
dump(pipeline_GB, "gradient_boosting_model.joblib")
gb_model = load("gradient_boosting_model.joblib") 
# Load saved model
#with open("gradient_boosting_model.pkl", "rb") as f:
    #gb_model = pickle.load(f)


# Accuracy Precision Recall F1 AUC-ROC - Gradient Boosting
print(classification_report(y_test, y_pred))
accuracy_GB = accuracy_score(y_test, y_pred)
precision_GB = precision_score(y_test, y_pred, average='weighted')
recall_GB = recall_score(y_test, y_pred, average='weighted')
f1_GB = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_GB = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_GB)
print("Precision:", precision_GB)
print("Recall:", recall_GB)
print("F1 Score:", f1_GB)
print("AUC-ROC Score:", auc_roc_GB)

# XG Boosting ML
# Declare target and features
X = data_attrition.drop(['Attrition'], axis=1)
y = data_attrition['Attrition']
# Split test and train data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=123)
# Create a pipeline:
pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy="not majority", random_state=123)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(
        n_estimators=300,       # number of boosting rounds
        learning_rate=0.1,      # step size shrinkage
        max_depth=5,            # depth of each tree
        subsample=0.5,          # fraction of samples per tree
        colsample_bytree=0.5,   # fraction of features per tree
        use_label_encoder=False,
        eval_metric='auc',  # 'auc' 'logloss'
        random_state=123
    ))
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Accuracy Precision Recall F1 AUC-ROC - XG Boosting
print(classification_report(y_test, y_pred))
accuracy_XGB = accuracy_score(y_test, y_pred)
precision_XGB = precision_score(y_test, y_pred, average='weighted')
recall_XGB = recall_score(y_test, y_pred, average='weighted')
f1_XGB = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_roc_XGB = auc(fpr, tpr)
print(cm)

print("Accuracy:", accuracy_XGB)
print("Precision:", precision_XGB)
print("Recall:", recall_XGB)
print("F1 Score:", f1_XGB)
print("AUC-ROC Score:", auc_roc_XGB)

# Performance rating Prediction
# set the data
perf_rating = data_attrition
perf_rating = perf_rating.drop(columns=["PerformanceRating"])
perf_rating["PerformanceRating"] = data["PerformanceRating"].values

# Performance Rating Prediction & Accuracy - Logistic Regression
# Declare target and features
A = perf_rating.drop(['PerformanceRating'], axis=1)
b = perf_rating['PerformanceRating']

# Train-test split
A_train, A_test, b_train, b_test = train_test_split(
    A, b, test_size=0.33, random_state=45
)

# Pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=100
    ))
])

# Train
pipeline.fit(A_train, b_train)

# Predict
b_pred = pipeline.predict(A_test)

# Metrics
print(classification_report(b_test, b_pred))

accuracy_LR = accuracy_score(b_test, b_pred)
precision_LR = precision_score(b_test, b_pred, average='weighted')
recall_LR = recall_score(b_test, b_pred, average='weighted')
f1_LR = f1_score(b_test, b_pred, average='weighted')
cm = confusion_matrix(b_test, b_pred)

print(cm)
print("Accuracy:", accuracy_LR)
print("Precision:", precision_LR)
print("Recall:", recall_LR)
print("F1 Score:", f1_LR)

# Pickle file
dump(pipeline, "logistic_regression_model.joblib")
lr_model = load("logistic_regression_model.joblib") 
# Load saved model
#with open("logistic_regression_model.pkl", "rb") as f:
   # lr_model = pickle.load(f)


# Back ground image
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# On choosing sidebar navigation
if side == "HOME":
    set_bg("D:/Project3/img8.jpg")
    st.markdown("""
    <h1 style="text-align: center;">EMPLOYEE ATTRITION</h1>
    <h2 style="text-align: center;"> WELCOME!</h2>   
    <h3 style="text-align: center;">This application uses employee data—such as experience, satisfaction levels, promotions, 
                performance, training sessions, and overtime—to predict both attrition risk and performance ratings. 
                By highlighting the key factors that influence why employees leave or how they perform, the system provides 
                HR with actionable insights to reduce turnover, improve employee engagement, 
                and strengthen workforce planning. </h3> 
    
    </style>               
    """,unsafe_allow_html=True)
elif side == "SUMMARY":
    set_bg("D:/Project3/img8.jpg")
    st.markdown("""<h3 style="text-align: center;">Attrition Prediction - Overall Performace of Models</h3>""",unsafe_allow_html=True)
    perfdata = {
    "ML": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "SVM",
        "Ada Boosting",
        "Gradient Boosting",
        "XG Boosting"
    ],
    "Accuracy": [accuracy_LR, accuracy_RF, accuracy_DT, accuracy_KNN, accuracy_SVM, accuracy_AB, accuracy_GB, accuracy_XGB],
    "Precision":[precision_LR, precision_RF, precision_DT, precision_KNN, precision_SVM, precision_AB, precision_GB, precision_XGB],
    "Recall":[recall_LR, recall_RF, recall_DT, recall_KNN, recall_SVM, recall_AB, recall_GB, recall_XGB],
    "F1":[f1_LR, f1_RF, f1_DT, f1_KNN, f1_SVM, f1_AB, f1_GB, f1_XGB],
    "AUC-ROC": [auc_roc_LR, auc_roc_RF, auc_roc_DT, auc_roc_KNN, auc_roc_SVM, auc_roc_AB, auc_roc_GB, auc_roc_XGB]
}

    ML_Perf = pd.DataFrame(perfdata)
    ML_Perf

elif side == "INSIGHTS - HR":
    set_bg("D:/Project3/img8.jpg")
    st.markdown("""<h3 style="text-align: center;">INSIGHTS - HR</h3>""",unsafe_allow_html=True)
    
    # KPI Metrics
    total_employees = data.shape[0]
    attrition_rate = data['Attrition'].value_counts(normalize=True)[1] * 100

    col1, col2 = st.columns(2)
    col1.metric("Total Employees", total_employees)
    col2.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")

    # Attrition Distribution by Satisfaction & Job Involvement

    st.subheader("Attrition Distribution by Satisfaction & Job Involvement")
    # ---- Encode Attrition safely ----
    data['Attrition'] = data['Attrition'].replace(
    {'Yes': 1, 'No': 0}
    ).astype(int)

    eda_columns = [
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'RelationshipSatisfaction',
    'JobInvolvement'
    ]

    # ---- Create figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, eda_columns):
        attrition_dist = pd.crosstab(
        data[col],
        data['Attrition'],
        normalize='index'
        ) * 100

        attrition_dist.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='coolwarm'
        )

        ax.set_title(f'Attrition Distribution by {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage')
        ax.legend(['No Attrition', 'Attrition'])

    # ---- Display in Streamlit ----
    st.pyplot(fig)

    # Attrition Heatmap by Performance Rating & Avg Satisfaction 
    ############################
    st.subheader("Attrition Heatmap by Performance Rating & Avg Satisfaction")

    # ---- Encode Attrition safely ----
    data['Attrition'] = data['Attrition'].replace(
    {'Yes': 1, 'No': 0}
    ).astype(int)

    # ---- Create Average Satisfaction Score ----
    satisfaction_cols = [
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'RelationshipSatisfaction',
    'JobInvolvement'
    ]

    data['AvgSatisfaction'] = data[satisfaction_cols].mean(axis=1)

    # ---- Bin Avg Satisfaction (for meaningful heatmap) ----
    data['AvgSatisfactionBin'] = pd.cut(
    data['AvgSatisfaction'],
    bins=[1, 2, 3, 4],
    labels=['Low', 'Medium', 'High']
    )

    # ---- Create pivot table (Attrition Rate %) ----
    heatmap_data = (
    data
    .groupby(['PerformanceRating', 'AvgSatisfactionBin'])['Attrition']
    .mean()
    .reset_index()
    )

    heatmap_pivot = heatmap_data.pivot(
    index='PerformanceRating',
    columns='AvgSatisfactionBin',
    values='Attrition'
    ) * 100

    # ---- Plot Heatmap ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
    heatmap_pivot,
    annot=True,
    fmt=".1f",
    cmap='Reds',
    linewidths=0.5,
    ax=ax
    )

    ax.set_title("Attrition Rate (%) by Performance Rating & Avg Satisfaction")
    ax.set_xlabel("Average Satisfaction Level")
    ax.set_ylabel("Performance Rating")

    # ---- Display in Streamlit ----
    st.pyplot(fig)

elif side == "ATTRITION PREDICTION":
    set_bg("D:/Project3/img8.jpg")
    st.markdown(
        """<h3 style="text-align: center;">ATTRITION PREDICTION</h3>""",
        unsafe_allow_html=True
    )
    
    st.write("Enter employee details below for prediction:")

    # Categorical Inputs (encoded during training)
    BusinessTravel = st.selectbox("Business Travel", [0, 1, 2])
    Department = st.selectbox("Department", [0, 1, 2])
    EducationField = st.selectbox("Education Field", [0, 1, 2, 3, 4, 5])
    Gender = st.selectbox("Gender", [0, 1])
    JobRole = st.selectbox("Job Role", list(range(9)))
    MaritalStatus = st.selectbox("Marital Status", [0, 1, 2])
    OverTime = st.selectbox("OverTime", [0, 1])

    # Rating / Satisfaction Fields
    Education = st.selectbox("Education", [1, 2, 3, 4, 5])
    JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
    StockOptionLevel = st.number_input("Stock Option Level", 0, 3)
    Total_Satisfaction = st.slider("Total Satisfaction (1–5)", 1.0, 5.0, 3.0)

    # Numerical Inputs
    Age = st.number_input("Age", 18, 60)
    DistanceFromHome = st.number_input("Distance From Home", 1, 30)
    MonthlyIncome = st.number_input("Monthly Income", 1000, 50000)
    NumCompaniesWorked = st.number_input("Num Companies Worked", 0, 10)
    PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 100)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40)
    TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10)
    YearsAtCompany = st.number_input("Years At Company", 0, 40)
    YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20)
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20)
    YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20)

    if st.button("Predict Attrition"):

        # Create dataframe IN THE EXACT ORDER
        user_input = pd.DataFrame([[
            BusinessTravel,
            Department,
            EducationField,
            Gender,
            JobRole,
            MaritalStatus,
            OverTime,
            Education,
            JobLevel,
            PerformanceRating,
            StockOptionLevel,
            Total_Satisfaction,
            Age,
            DistanceFromHome,
            MonthlyIncome,
            NumCompaniesWorked,
            PercentSalaryHike,
            TotalWorkingYears,
            TrainingTimesLastYear,
            YearsAtCompany,
            YearsInCurrentRole,
            YearsSinceLastPromotion,
            YearsWithCurrManager
        ]], columns=[
            'BusinessTravel',
            'Department',
            'EducationField',
            'Gender',
            'JobRole',
            'MaritalStatus',
            'OverTime',
            'Education',
            'JobLevel',
            'PerformanceRating',
            'StockOptionLevel',
            'Total_Satisfaction',
            'Age',
            'DistanceFromHome',
            'MonthlyIncome',
            'NumCompaniesWorked',
            'PercentSalaryHike',
            'TotalWorkingYears',
            'TrainingTimesLastYear',
            'YearsAtCompany',
            'YearsInCurrentRole',
            'YearsSinceLastPromotion',
            'YearsWithCurrManager'
        ])

        # Predict
        prediction = gb_model.predict(user_input)[0]
        probability = gb_model.predict_proba(user_input)[0][1]

        if prediction == 1:
            st.markdown(
                f"""
                <div style="background-color: white; padding: 15px; border-radius: 10px;
                    border: 2px solid #ff4b4b; font-size: 18px;">
                <b>⚠️ Employee LIKELY TO LEAVE.</b><br>
                Attrition Risk: <b>{probability:.2f}</b>
                </div>
                """,
                unsafe_allow_html=True
                    )
        else:
            st.markdown(
                f"""
                <div style="background-color: white; padding: 15px; border-radius: 10px;
                    border: 2px solid #4caf50; font-size: 18px;">
                    <b>✔️ Employee LIKELY TO STAY.</b><br>
                    Attrition Risk: <b>{probability:.2f}</b>
                    </div>
                            """,
                unsafe_allow_html=True
                 )
elif side == "RATINGS PREDICTION":
    set_bg("D:/Project3/img8.jpg")

    st.markdown(
        """<h3 style="text-align: center;">PERFORMANCE RATING PREDICTION</h3>""",
        unsafe_allow_html=True
    )

    st.write("Enter employee details below for performance rating prediction:")

    # ---- Attrition (REQUIRED FEATURE) ----
    Attrition = st.selectbox(
        "Attrition Status",
        options=[0, 1],
        format_func=lambda x: "No Attrition" if x == 0 else "Attrition"
    )

    # ---- Categorical Inputs (Label Encoded) ----
    BusinessTravel = st.selectbox("Business Travel", [0, 1, 2])
    Department = st.selectbox("Department", [0, 1, 2])
    EducationField = st.selectbox("Education Field", [0, 1, 2, 3, 4, 5])
    Gender = st.selectbox("Gender", [0, 1])
    JobRole = st.selectbox("Job Role", list(range(9)))
    MaritalStatus = st.selectbox("Marital Status", [0, 1, 2])
    OverTime = st.selectbox("OverTime", [0, 1])

    # ---- Ratings / Job Information ----
    Education = st.selectbox("Education", [1, 2, 3, 4, 5])
    JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    StockOptionLevel = st.number_input("Stock Option Level", 0, 3)
    Total_Satisfaction = st.slider("Total Satisfaction (1–5)", 1.0, 5.0, 3.0)

    # ---- Numeric Inputs ----
    Age = st.number_input("Age", 18, 60)
    DistanceFromHome = st.number_input("Distance From Home", 1, 30)
    MonthlyIncome = st.number_input("Monthly Income", 1000, 50000)
    NumCompaniesWorked = st.number_input("Num Companies Worked", 0, 10)
    PercentSalaryHike = st.number_input("Percent Salary Hike", 0, 100)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40)
    TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10)
    YearsAtCompany = st.number_input("Years At Company", 0, 40)
    YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20)
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20)
    YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20)

    if st.button("Predict Performance Rating"):

        # ---- Create input dataframe (MUST MATCH TRAINING ORDER) ----
        user_input = pd.DataFrame([[ 
            Attrition,
            BusinessTravel,
            Department,
            EducationField,
            Gender,
            JobRole,
            MaritalStatus,
            OverTime,
            Education,
            JobLevel,
            StockOptionLevel,
            Total_Satisfaction,
            Age,
            DistanceFromHome,
            MonthlyIncome,
            NumCompaniesWorked,
            PercentSalaryHike,
            TotalWorkingYears,
            TrainingTimesLastYear,
            YearsAtCompany,
            YearsInCurrentRole,
            YearsSinceLastPromotion,
            YearsWithCurrManager
        ]], columns=[
            'Attrition',
            'BusinessTravel',
            'Department',
            'EducationField',
            'Gender',
            'JobRole',
            'MaritalStatus',
            'OverTime',
            'Education',
            'JobLevel',
            'StockOptionLevel',
            'Total_Satisfaction',
            'Age',
            'DistanceFromHome',
            'MonthlyIncome',
            'NumCompaniesWorked',
            'PercentSalaryHike',
            'TotalWorkingYears',
            'TrainingTimesLastYear',
            'YearsAtCompany',
            'YearsInCurrentRole',
            'YearsSinceLastPromotion',
            'YearsWithCurrManager'
        ])

        # ---- Predict ----
        prediction = lr_model.predict(user_input)[0]
        probabilities = lr_model.predict_proba(user_input)[0]
        confidence = probabilities.max()

        # ---- Display Result ----
        st.markdown(
            f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px;
                border: 2px solid #4caf50; font-size: 18px;">
            <b>Predicted Performance Rating:</b> ⭐ <b>{prediction}</b><br>
            </div>
            """,
            unsafe_allow_html=True
        )