# Dawood_Khan

**House Price Prediction Using Zameen.com Data**
  This project predicts house prices based on various features provided in the dataset. It demonstrates data cleaning, feature encoding, and model training using machine learning           algorithms.

**Features of the Project**
> Data cleaning and preprocessing
> Handling missing values
> Encoding categorical data with get_dummies
> Model training and evaluation
> Comparison of LinearRegression and DecisionTreeRegressor for predictions
> Dataset

The dataset was collected from Zameen.com, featuring details about properties such as location, size, and other relevant attributes.

**Steps in the Project**
Data Cleaning:
  Dropped rows with null values to ensure data quality.
Feature Encoding:
  Used get_dummies to convert categorical features into numerical values.
Model Training:
  Initially trained the data on LinearRegression, but the results were suboptimal.
  Switched to DecisionTreeRegressor, which improved the performance.
**Results:**
  Linear Regression: The model did not perform well due to the complexity of the dataset and non-linearity in relationships.
  Decision Tree Regressor: Provided significantly better predictions for house prices.
