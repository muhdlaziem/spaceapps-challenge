import pandas as pd
import numpy as np

original_data = pd.read_csv("meteorite-landings.csv") # We first put the data into a dataframe called original data
original_data.to_csv('Original_data.csv')
print(original_data)
'''
We then check the total number of missing values in each column, just to understand the data better
'''
#print(original_data.isna().sum())

'''Assigning any incomplete rows into another dataframe called incomplete_data '''
incomplete_data = original_data[original_data.isnull().values.any(axis=1)]
#print(incomplete_data)
#print(incomplete_data.isna().sum())

'''Assigning all complete data to another dataframe called complete_data  '''
complete_data = original_data.dropna()
complete_data.to_csv('complete_data.csv')
print(complete_data)
#print(complete_data.isna().sum())


'''This part here is to read, check and make sure all the data has been accounted for 
and assigned to the right dataframe'''

print(original_data.describe())
print(incomplete_data.describe())
print(complete_data.describe())

print(original_data)
print(incomplete_data)
print(complete_data)

''' The next step is to purposely remove some of the data or values in the complete set
before choosing some of the training model
The line below here allow us to omit 20% of the values randomly '''


data_mask = np.random.choice([True, False], size=complete_data.shape, p=[.2,.8])
omit_data=complete_data.mask(data_mask)
omit_data.to_csv('Omit_data.csv')
print(omit_data)
print(omit_data.isna().sum())



