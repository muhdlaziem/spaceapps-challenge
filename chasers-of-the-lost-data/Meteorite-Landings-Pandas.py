import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

original_data = pd.read_csv("meteorite-landings.csv",
                            index_col=0)  # We first put the data into a dataframe called original data
original_data.to_csv('Original_data.csv')
print(original_data)
'''Note that we can use print(original_data.shape) if we just want to see how many rows and columns are there '''
'''
We then check the total number of missing values in each column, just to understand the data better
'''
# print(original_data.isna().sum())

'''Assigning any incomplete rows into another dataframe called incomplete_data '''
incomplete_data = original_data[original_data.isnull().values.any(axis=1)]
# print(incomplete_data)
# print(incomplete_data.isna().sum())

'''Assigning all complete data to another dataframe called complete_data  '''
complete_data = original_data.dropna()
complete_data.to_csv('complete_data.csv')
print(complete_data)
# print(complete_data.isna().sum())


'''This part here is to read, check and make sure all the data has been accounted for 
and assigned to the right dataframe'''

print(original_data.describe())
print(incomplete_data.describe())
print(complete_data.describe())

print(original_data)
print(incomplete_data)
print(complete_data)

''' 
The next step is to purposely remove some of the data or values in the complete set
before choosing some of the training model
The line below here allow us to omit 20% of the values randomly 
'''

data_mask = np.random.choice([True, False], size=complete_data.shape, p=[.2, .8])
omit_data = complete_data.mask(data_mask)
omit_data.to_csv('Omit_data.csv')
print(omit_data)
print(omit_data.isna().sum())

''' trying to do the same thing above, except this time I tried to target a specific columns
    before that, check for any correlation. It can be either pearson, kendall or even spearman

    This is how to select specific column
    spec_col= omit_data.iloc[:,[5]] # selecting column 5. Note that it started from 0 
'''
print(omit_data.corr(method='pearson'))
plt.matshow(omit_data.corr(method='pearson'))
plt.show()  # simply wanted to show correlation matrix

'''

this part kinda don't make sense since the correlation is too low, and not correct
therefore the regression line form is inaccurate

'''
x = complete_data.iloc[:, [5]].values.reshape(-1, 1)  # select column 5
y = complete_data.iloc[:, [3]].values.reshape(-1, 1)  # select column 3
imputation_model = LinearRegression()
imputation_model.fit(x, y)

# spec_col.to_csv('Try.csv')  # Not needed, just for manual validation and check

y_predict = imputation_model.predict(x)
print("\n------Result of prediction------\n")
print(y_predict)
print("\n------Result of prediction------\n")
plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()  # can omit this out if not needed

''' 
    this part here is using mean imputations (a type of statistical imputations)
    the library entered here is for RMSE calculations
'''

omit_data["mass"].fillna(omit_data.groupby("name")["mass"].transform("mean"), inplace=True)
omit_data["mass"].fillna(omit_data["mass"].mean(), inplace=True)

print(omit_data.isna().sum())

result1 = omit_data
result1.to_csv("Result1.csv")

val_actual = complete_data.iloc[:, [3]].values.reshape(-1, 1)  # select column 5
val_predict = omit_data.iloc[:, [3]].values.reshape(-1, 1)  # select column 5

'''
    This part is wrong due to lack of insight in comparison and calculating error
'''
rms = sqrt(mean_squared_error(val_actual, val_predict))

print("\nThe value of RMSE : " + str(rms))



