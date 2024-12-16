import pickle
import pandas as pd
import numpy as np

# Load the pickled model
with open('logistic_regression_model.sav', 'rb') as file:
    model = pickle.load(file)

# Load the test dataset
dataset = pd.read_csv('synthetic cattle_dataset.csv')

# Preprocess the dataset
dataset = dataset.drop('faecal_consistency', axis=1)
dataset = dataset.drop('breed_type', axis=1)
dataset.dropna(inplace=True)
dataset = dataset.replace({'healthy': 1, 'unhealthy': 0})

# Split the dataset into features and target variables
X_test = dataset.drop('health_status', axis=1)
y_test = dataset['health_status']

# Evaluate the model's performance
performance = model.score(X_test, y_test)

# Print the performance metric
print("Model performance:", performance)

# Predict with the given values

new_data = np.array([[38.5, 20.2, 16.0, 5004.0, 4.6, 5.0, 55.0, 2.5, 12.5, 4.4, 2.0]])


prediction = model.predict(new_data)
# note: predicted value should be healthy
# Map the predicted value to 'healthy' or 'unhealthy'
result = 'healthy' if prediction[0] == 1 else 'unhealthy'
print("Predicted value:", result)
