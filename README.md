# CowCare - AI-Powered Cow Disease Prediction and Advice System

## Overview
CowCare is a machine learning-based application designed to predict cow diseases and provide actionable advice based on user inputs. The system aims to assist farmers and livestock owners in maintaining the health of their cattle by offering early warnings and expert recommendations.

## Features
- **Disease Prediction**: Uses machine learning algorithms to predict potential cow diseases based on symptoms entered by the user.
- **Expert Advice**: Provides tailored advice to manage or prevent the identified disease.
- **Interactive Web Interface**: A user-friendly website to input data and view predictions and recommendations.

## Approach
### First Module
- In this module, we calculate the **Performance Score** of six different machine learning models.  
- The model with the highest performance score is chosen to predict the **Abnormality** in cattle.  
- Once abnormalities are detected, the system proceeds to the second module.

### Second Module
- For disease prediction, we use four different machine learning models.  
- The **common prediction** from these models is taken as the output.  
- In case of unique predictions, the **Random Forest Algorithm** is used to determine the final prediction.

## Tech Stack
- **Frontend**: Interactive website built with python.  
- **Backend**: Machine learning model implemented using Python and integrated with the web application.  

## How It Works
1. **Input Symptoms**: Users enter symptoms and other relevant data through an interactive web interface.  
2. **Prediction**: The system processes the input using a trained machine learning model to:  
   - Detect abnormalities in cattle (First Module).  
   - Predict potential diseases (Second Module).  
3. **Advice**: Based on the prediction, actionable advice is displayed to the user.

