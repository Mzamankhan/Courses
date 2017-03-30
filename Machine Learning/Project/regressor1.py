'''
Apply Nearest Neighbor Regressor to Encoded data of Housing price
'''

from __future__ import print_function
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#Set Threshold for correlation
THRESHOLD = 0.1

numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal','MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
ordinal = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

# Read in train and test data (Categorical data are encoded)
num_train_data = pd.read_csv('new_train.csv', index_col = 0)
num_test_data = pd.read_csv('test_data_no_na.csv', index_col = 0)
num_train_data['log_sale_price'] = np.log10(num_train_data['SalePrice'])
num_train_data_wo_sale = num_train_data.drop('SalePrice',1)
num_train_data_wo_sale = num_train_data_wo_sale.drop('log_sale_price',1)
#Read original train_data
house_data = pd.read_csv( 'train.csv', index_col = 0, na_filter = False)

# Read in combined data (Categorical data rae not encoded)
combined = pd.read_csv('combined.csv', index_col = 0, na_filter=False)
#Read Correlations file
correlation = pd.read_csv('cor_cofs.csv', index_col = 0)
#print(list(correlation))
#Filter Attributes based on correlation
corr_cols = correlation[correlation['Corr']>THRESHOLD]
high_corr_cols = corr_cols['Attr'].tolist()

#Normalize train data
attr = numeric+ordinal
num_or = numeric+ordinal
min_train_data = num_train_data[num_or].min()
max_train_data = num_train_data[num_or].max()
#print(min_numeric_data['LotFrontage'])
#house_data1 = house_data
#combined[num_or] = (combined[num_or] - min_train_data) /(max_train_data-min_train_data)
num_train_data[num_or] = (num_train_data[num_or] - min_train_data) /(max_train_data-min_train_data)
num_test_data[num_or] = (num_test_data[num_or] - min_train_data) /(max_train_data-min_train_data)
'''
#Dummify
dummies = pd.get_dummies( combined, columns = nominal )
dummy_train_data_w_sale = dummies[ :house_data.shape[0]]
dummy_test_data = dummies[house_data.shape[0]:]
dummy_train_data_w_sale['log_sale_price'] = np.log10(dummy_train_data_w_sale['SalePrice'])
dummy_train_data_wo_sale = dummy_train_data_w_sale.drop('SalePrice',1)
dummy_train_data_wo_sale = dummy_train_data_wo_sale.drop('log_sale_price',1)
'''
#Do K fold Validation on regressor
folds = 5
total_error = 0
tot_preds = []
tot_len =0
#print("Total number of example: ")
#print(dummy_train_data_w_sale.shape[0],end="\n")
#for n in range(1,11):
n=1
#print("For neighbors: ", end=" ")
#print(n, end="\n")
reg = LinearRegression()#GradientBoostingRegressor()#LinearRegression()#GradientBoostingRegressor()#RandomForestRegressor()#GradientBoostingRegressor()##SGDRegressor()#RandomForestRegressor()#SGDRegressor()##Lasso(fit_intercept=True) #AdaBoostRegressor()#Lasso(fit_intercept=True)#, weights='distance')
kf = KFold(n_splits = folds)
kf.get_n_splits(num_train_data)
for train_index, test_index in kf.split(num_train_data):
	X_train, X_test =num_train_data_wo_sale.iloc[train_index][attr], num_train_data_wo_sale.iloc[test_index][attr] #dummy_train_data_wo_sale.iloc[train_index][high_corr_cols], dummy_train_data_wo_sale.iloc[test_index][high_corr_cols]
	y_train, y_test = num_train_data.iloc[train_index]['log_sale_price'], num_train_data.iloc[test_index]['log_sale_price']
	# Fit nn regressor on training data , train data contains SalePrice (Should change)
	#neigh = KNeighborsRegressor(n_neighbors = n)#, weights='distance')
	reg.fit(X_train,y_train)#,y_train)
	pred = reg.predict(X_test)
	error = abs(pred - y_test)
	model_error = sum(error)/len(error)
	total_error += model_error
	print('Absolute Mean Error Model: ')
	print(model_error)
	#print("length of pred: ")
	#print(len(pred), end="\n")
	#tot_preds.append(pred.tolist())
	tot_preds = tot_preds+pred.tolist()
	tot_len += len(pred)
	#print(tot_preds.shape)
final_error = total_error/folds
print('Average Absolute Mean Error:', end=" ")
print(final_error, end="\n")



#dummy_test_data=dummy_test_data.drop('SalePrice',1)
print(num_train_data_wo_sale.head(),end="\n")

num_test_data = num_test_data.drop('SalePrice',1)
print(num_test_data.shape,end="\n")
test_pred = reg.predict(num_test_data[attr])
num_test_data['SalePrice'] = test_pred
num_test_data['SalePrice'] = pow(10,num_test_data['SalePrice'])

num_train_data["Predicted_log_SalePrice"] = tot_preds
num_train_data['Predicted_SalePrice'] = pow(10,num_train_data['Predicted_log_SalePrice'])
#print(test_pred)
#num_test_data.to_csv("rf_allcols_submission.csv")
#Append prediction to SalePrice dataframe
num_train_data["Predicted_log_SalePrice"] = tot_preds
#plt.scatter( num_train_data['log_sale_price'], num_train_data['Predicted_log_SalePrice'], s = 80, alpha = 0.4 )
#plt.show()

correlation = num_train_data["Predicted_log_SalePrice"].corr(num_train_data['log_sale_price'])
print (correlation)

plt.scatter( num_train_data['SalePrice'], num_train_data['Predicted_SalePrice'], s = 80, alpha = 0.4 )
    # plt.title( 'Correlation: {} and r^2: {}'.format( predictions_df['SalePrice'].corr( predictions_df['fold_predictions'] ), r2_score( predictions_df['SalePrice'], predictions_df['fold_predictions'] ) ) )
plt.axis( [0, max( num_train_data['SalePrice'] ), 0, max( num_train_data['SalePrice'] )] )
plt.xlabel( 'Actual Sale Price' )
plt.ylabel( 'Predicted Sale Price' )
plt.title( 'Actual Vs. Predicted Sale Price - Correlation {}'.format( correlation ) ) 
plt.show()
