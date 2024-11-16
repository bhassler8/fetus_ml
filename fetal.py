# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 

# Display an image and description
st.image('fetal_health_image.gif', width = 400)
st.write("Utilize our advanced Machine Learning application to predict fetal health classification")
st.write("Please upload a .csv file on the sidebar with the specification shown in the example")

# Load the pre-trained model from the pickle files
dt_pickle = open('dt_fetus.pickle', 'rb') #read bytes, we aren't trying to write them
fetal_dt = pickle.load(dt_pickle)
dt_pickle.close()

rf_pickle = open('rf_fetus.pickle', 'rb')
fetal_rf = pickle.load(rf_pickle)
rf_pickle.close()

ada_pickle = open('ada_fetus.pickle', 'rb')
fetal_ada = pickle.load(ada_pickle)
ada_pickle.close()

sv_pickle = open('sv_fetus.pickle', 'rb')
fetal_sv = pickle.load(sv_pickle)
sv_pickle.close()

default_df = pd.read_csv('fetal_health.csv')
default_df = default_df.drop('fetal_health', axis=1)

st.sidebar.header('Fetal Health Features Input')
user_csv = st.sidebar.file_uploader('')
st.sidebar.title('Sample Data for Upload')
st.sidebar.dataframe(default_df.head())
st.sidebar.write("Make sure your file has the same columns and data types as shown above")

model = st.sidebar.radio("Choose your Model for Prediction", ('Decision Tree', 'Random Forest', 'AdaBoost', 'Soft Voting'))

#used chat gpt to help with code syntax on how to add colors to columns, uses a function

def color(val):
    color_map = {
        'Normal': 'background-color: lime;',
        'Suspect': 'background-color: yellow;',
        'Pathological': 'background-color: orange;'
    }
    return color_map.get(val, '')

if user_csv is not None:

    user_df = pd.read_csv(user_csv)
    encoded_user_df = pd.get_dummies(user_df)

    if model == 'Decision Tree':
        dt_prediction = fetal_dt.predict(encoded_user_df)
        user_df['Predicted Fetal Health'] = dt_prediction
        dt_probability = fetal_dt.predict_proba(encoded_user_df)
        max_prob = np.max(dt_probability, axis = 1)
        user_df['Prediction Probability %'] = max_prob * 100
        colored_df = user_df.style.applymap(color, subset=['Predicted Fetal Health'])
        st.write(colored_df)

        st.subheader("Prediction Performance")
        tab1, tab2, tab3, = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('fetus_dt_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetus_dt_confusion_matrix.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetus_dt_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support")

    if model == 'Random Forest':
        rf_prediction = fetal_rf.predict(encoded_user_df)
        user_df['Predicted Fetal Health'] = rf_prediction
        rf_probability = fetal_rf.predict_proba(encoded_user_df)
        max_prob = np.max(rf_probability, axis = 1)
        user_df['Prediction Probability %'] = max_prob * 100
        colored_df = user_df.style.applymap(color, subset=['Predicted Fetal Health'])
        st.write(colored_df)

        st.subheader("Prediction Performance")
        tab1, tab2, tab3, = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('fetus_rf_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetus_rf_confusion_matrix.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetus_rf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support")

    if model == 'AdaBoost':
        ada_prediction = fetal_ada.predict(encoded_user_df)
        user_df['Predicted Fetal Health'] = ada_prediction
        ada_probability = fetal_ada.predict_proba(encoded_user_df)
        max_prob = np.max(ada_probability, axis = 1)
        user_df['Prediction Probability %'] = max_prob * 100
        colored_df = user_df.style.applymap(color, subset=['Predicted Fetal Health'])
        st.write(colored_df)

        st.subheader("Prediction Performance")
        tab1, tab2, tab3, = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('fetus_ada_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetus_ada_confusion_matrix.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetus_ada_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support")

    if model == 'Soft Voting':
        sv_prediction = fetal_sv.predict(encoded_user_df)
        user_df['Predicted Fetal Health'] = sv_prediction
        sv_probability = fetal_sv.predict_proba(encoded_user_df)
        max_prob = np.max(sv_probability, axis = 1)
        user_df['Prediction Probability %'] = max_prob * 100
        colored_df = user_df.style.applymap(color, subset=['Predicted Fetal Health']).format({'Predction Probability %': '{:.2f}'})
        st.write(colored_df)

        st.subheader("Prediction Performance")
        tab1, tab2, tab3, = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('fetus_sv_feature_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 2: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('fetus_sv_confusion_matrix.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 3: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('fetus_sv_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support")