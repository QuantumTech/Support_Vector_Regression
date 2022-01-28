# Support Vector Regression (SVR)

# Assumptions of linear regression:
# Ensure all these assumptions are TRUE!
# Linearity
# Homoscedasticity
# Multivariate normality
# Independence of errors
# Lack of multicollinearity

# Dummy variables:
# Find all the different categorical data within the dataset
# Turn String values into binary

# Dummy variables traps

# Pvalues! (probability value)
# Statistical significance:
# 1) Flip a coin (two possible outcomes)

# 5 Methods of building models:
# 1) All-in (step wise regression) = Use of all variables

# 2) Backward elimination (step wise regression) =
# Step 1: select a significance level to stay in the model (e.g SL = 0.05)
# Step 2: Fit the full model with all possible predictors
# Step 3: Consider the predictor with the highest P-Value. If P > SL (significance level), go to step 4, otherwise go to FIN (model is ready)
# Step 4: Remove the predictor
# Step 5: Fit model without this variable
# Return back to Step 3

# 3) Forward selection (step wise regression)
# Step 1: Select a significance level to enter the model (e.g SL = 0.05)
# Step 2: Fit all simple regression models 'Y~Xn' select the one with the lowest P-value
# Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
# Step 4: Consider the predictor with the lowest P-value. If P<SL, go to Step 3, otherwise go to FIN

# 4) Bidirectional elimination (step wise regression)
# Step 1: Select a significance level to enter and to stay in the model
# Step 2: Perform the next step of Forward selection
# Step 3: Perform ALL steps of Backward elimination
# Step 4: No new variables can enter and no old variables can exit

# 5) Score comparison
# Step 1: Select a criterion of goodness of fit
# Step 2: Construct All possible Regression Models
# Step 3: Select the one with the best criterion

# BACKWARD ELIMINATION will be used for this project

# Support Vector Regression (SVR) uses the absolute insensitive tube.
# The dots outside the absolute insensitive tube are the support vector regressions.
# Notes to self: Look into kernal SVR (3D graph.)
# Notes to self: SVM Kernal!

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# IMPORTING THE LIBRARIES

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# IMPORTING THE DATASET

# Data set is creating the data frame of the '50_Startups.csv' file
# Features (independent variables) = The columns the predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

# 'X = dataset.iloc[:, 1:-1].values' (This will take the values from the second column (index 1) to the second from last on)

# 'y = dataset.iloc[:, -1].values' (NOTICE! .iloc[all the rows, only the last column])

# 'X' (This contains the 'x' axis in a vertical 2D array)
# 'y' (This contains the 'y' axis in a 1D horizontal vector)

# 'y = y.reshape(len(y),1)' ('y' must be reshaped into a vertical 2D array. (The number of rows, columns(1)))

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape(len(y),1)
print(y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# FEATURE SCALING

# Why feature scale after splitting the data? = Because the 'test set' is a brand new set that is not supposed to be part of the training set

# This stage is not essential!

# Regression: simple linear regression, multiple linear regression, polynomial linear regression

# There are two feature scaling techniques STANDARDISATION & NORMALISATION

# Normalisation: All the values of the features will be between 0 and 1

# Normalisation: Recommended when you have a normal distribution in most of your features.

# Standardisation: All the values of the features will be between -3 and 3

# Standardisation: Works well all the time (RECOMMENDED)

# Implicit equation; therefore I must use Feature scaling

# 'sc_X = StandardScaler()'  Scaler of 'x'
# 'sc_y = StandardScaler()' Dependent variable 'y'

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing.

# TRAINING THE SVR MODEL ON THE WHOLE DATASET

# (SVM Kernel Function; Gaussian Radial Basis Function (RBF))
# 'regressor.fit(X, y)' = regressor on the dataset

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, np.ravel(y))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Support Vector Regression (SVR)

# PREDICTING A NEW RESULT

# Reverse the scaling
# This contains a 2D array with a single value
# Note we are getting the level 6.5 from the csv file
# It will return the predicted salary in the scale of 'y' (Meaning the variables 'y' from the feature scaling phase.)

# Inverse transformation is required to predict in the original format otherwise it will return in standard scalar format

y_pred = sc_y.inverse_transform([regressor.predict(sc_X.transform([[6.5]]))])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Support Vector Regression (SVR)

# VISUALISING THE SVR RESULTS

# Reminder 'plt' is shorthand for MatPlotLib
# Coordinates for the scatter plot, using the positions 'x' and 'y'
# The inverse transform method is being used.
# Showing the predicted results on the regression curve.

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform([regressor.predict(X)]).reshape(10,1), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# As the original code block was showing errors that it needed a 2D array input, 'm' was created to reshape it
# This works with the original prediction code
""" m = regressor.predict(X)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(m.reshape(len(m),1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() """

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Support Vector Regression (SVR)

# VISUALISING THE SVR RESULTS (FOR HIGHER RESOLUTION AND SMOOTHER CURVE)

# Very small increments of '0.1' connecting each line, which creates a smooth curve.
# Reversing the scale of 'x' then reversing the scale of 'y'

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform([regressor.predict(sc_X.transform(X_grid))]).reshape(90,1), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# This works with the original prediction code
""" X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
k = regressor.predict(sc_X.transform(X_grid))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(k.reshape(len(k),1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() """