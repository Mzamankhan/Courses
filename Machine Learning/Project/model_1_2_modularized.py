'''
Created on Mar 17, 2017


Binning the data into different number of bins
and training regression models and checking their accuracy.
'''
import os
import sys

from matplotlib.lines import Line2D
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import helpers
import matplotlib.pyplot as plt
from multi_column_label_encoder import MultiColumnLabelEncoder
import numpy as np
import pandas as pd


converters = {
              'MasVnrType': lambda x: None if x == 'NA' else x,
              'Electrical': lambda x: None if x == 'NA' else x,
              'LotFrontage': lambda x: None if x == 'NA' else int( x ),
              'MasVnrArea': lambda x: 0 if x == 'NA' else int( x ),
              'GarageYrBlt': lambda x: 0 if x == 'NA' else int( x ),

              'BsmtFinSF1': lambda x: None if x == 'NA' else int( x ),
              'BsmtFinSF2': lambda x: None if x == 'NA' else int( x ),
              'BsmtUnfSF': lambda x: None if x == 'NA' else int( x ),
              'TotalBsmtSF': lambda x: None if x == 'NA' else int( x ),
              'BsmtFullBath': lambda x: None if x == 'NA' else int( x ),
              'BsmtHalfBath': lambda x: None if x == 'NA' else int( x ),
              'GarageCars': lambda x: None if x == 'NA' else int( x ),
              'GarageArea': lambda x: None if x == 'NA' else int( x ),
              }

numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
ordinal = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

reduced_numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch']
reduced_nominal = ['MSSubClass', 'MSZoning', 'LotShape', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'SaleType', 'SaleCondition']
reduced_ordinal = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']

reduced_numeric2 = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']
reduced_ordinal2 = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars']

# attributes_dt = reduced_nominal + reduced_nominal + reduced_ordinal
# attributes_reg = numeric
attributes_dt = reduced_numeric2 + reduced_ordinal2 + reduced_nominal
attributes_reg = reduced_numeric2 + reduced_ordinal2


house_data = pd.read_csv( 'train.csv', index_col = 0, na_filter = False, converters = converters )

test_data = pd.read_csv( 'test.csv', index_col = 0, na_filter = False, converters = converters )

# Filling missing values
house_data[['MasVnrArea']] = house_data[['MasVnrArea']].fillna( house_data[['MasVnrArea']].mean() )
house_data[['LotFrontage']] = house_data[['LotFrontage']].fillna( house_data[['LotFrontage']].mean() )
house_data[['MasVnrType']] = house_data[['MasVnrType']].fillna( house_data[['MasVnrType']].mode() )
house_data[['Electrical']] = house_data[['Electrical']].fillna( house_data[['Electrical']].mode() )

test_data[['MasVnrArea']] = test_data[['MasVnrArea']].fillna( house_data[['MasVnrArea']].mean() )
test_data[['LotFrontage']] = test_data[['LotFrontage']].fillna( house_data[['LotFrontage']].mean() )
test_data[['MasVnrType']] = test_data[['MasVnrType']].fillna( house_data[['MasVnrType']].mode() )
test_data[['Electrical']] = test_data[['Electrical']].fillna( house_data[['Electrical']].mode() )
test_data[['BsmtFinSF1']] = test_data[['BsmtFinSF1']].fillna( house_data[['BsmtFinSF1']].mean() )
test_data[['BsmtFinSF2']] = test_data[['BsmtFinSF2']].fillna( house_data[['BsmtFinSF2']].mean() )
test_data[['BsmtFinSF1']] = test_data[['BsmtFinSF1']].fillna( house_data[['BsmtFinSF1']].mean() )
test_data[['BsmtUnfSF']] = test_data[['BsmtUnfSF']].fillna( house_data[['BsmtUnfSF']].mean() )
test_data[['TotalBsmtSF']] = test_data[['TotalBsmtSF']].fillna( house_data[['TotalBsmtSF']].mean() )
test_data[['BsmtFullBath']] = test_data[['BsmtFullBath']].fillna( 0 )  # .fillna( house_data[['BsmtFullBath']].mode() )  #
test_data[['BsmtHalfBath']] = test_data[['BsmtHalfBath']].fillna( 0 )  # .fillna( house_data[['BsmtHalfBath']].mode() )  #
test_data[['GarageCars']] = test_data[['GarageCars']].fillna( 2 )  # .fillna( house_data[['GarageCars']].mode() )  #
test_data[['GarageArea']] = test_data[['GarageArea']].fillna( house_data[['GarageArea']].mean() )

# print test_data.shape
zero_price = [1 for i in range( test_data.shape[0] )]
test_data['SalePrice'] = zero_price
# print test_data.shape

combined = pd.concat( [house_data, test_data] )

# print ( combined.shape )

# # print ( combined.tail() )
# changed = pd.get_dummies( combined, columns = nominal )
# # print ( dummies.head() )
# # print ( dummies.shape )
# new_attributes = changed.columns.values
# index = np.argwhere( new_attributes == 'SalePrice' )
# new_attributes = np.delete( new_attributes, index )
# attributes_dt = new_attributes
# attributes_reg = new_attributes


changed = MultiColumnLabelEncoder( columns = nominal ).fit_transform( combined )
# print ( changed.shape )

# print ( changed.tail() )
# changed['log_sale_price'] = np.log10( changed['SalePrice'] )
predict_column = 'SalePrice'  # 'log_sale_price'  #


new_train = changed[:][ :house_data.shape[0]]
new_test = changed[:][house_data.shape[0]:]

# new_test.to_csv( '++test_set_3_19.csv' )
# exit()

new_test = new_test.drop( 'SalePrice', 1 )
# new_test = new_test.drop( predict_column, 1 )


# print new_train.shape
# print new_test.shape

# new_train.to_csv( 'new_train.csv' )
# new_test.to_csv( 'new_test.csv' )

plot_dir = os.path.join( os.getcwd(), 'bin_distributions' )
if not os.path.exists( plot_dir ):
    os.mkdir( plot_dir )

sorted_hp = sorted( new_train[predict_column] )
samples = house_data.shape[0]
folds = 5

decision_tree_bin_errors = []
combined_bin_min_errors = []
combined_bin_model_errors = []
combined_bin_no_model_errors = []
combined_bin_no_model_min_errors = []
reg_models = []

full_bin_tot_min_err = []
full_bin_tot_model_err = []
full_bin_tot_class_pred_mean_err = []
full_bin_tot_min_pred_mean_err = []

# classifier = tree.DecisionTreeClassifier
# dt = classifier()
classifire = tree.DecisionTreeClassifier( max_depth = 10 )
#classifire = RandomForestClassifier( n_estimators = 20 )
#classifire = GradientBoostingClassifier()

regressor = linear_model.LinearRegression()
#regressor = linear_model.Lasso( alpha = 0.1 )
#regressor = RandomForestRegressor( n_estimators = 20 )
#regressor = GradientBoostingRegressor()

# Try different numbers of bins
for bins in range( 2, 3 ):

    bins, bin_data, boundaries = helpers.create_equal_frequency_bins( samples, bins, new_train, sorted_hp, predict_column )

    # print ( tot )
    print ( 'Actual bins created  : {}'.format( bins ) )
    print ( boundaries )

    # bin_labels = pd.Series( [helpers.sale_price_to_bin( boundaries, sp ) for sp in new_train[predict_column]] )
    bin_labels = pd.Series( [helpers.sale_price_to_bin( boundaries, sp ) for sp in new_train[predict_column]], range( 1, len( new_train ) + 1 ) )

    # To create a data set with bin labels
#     new_train['Bin'] = bin_labels
#     house_data['Bin'] = bin_labels
#     # print new_train.tail()
#     new_train.to_csv( '++nominal_dataset_encoded.csv', columns = nominal + ['Bin'] )
#     house_data.to_csv( '++nominal_dataset_original.csv', columns = nominal + ['Bin'] )
#     exit()

    # print ( bin_labels[:10] )

    partition_bin = helpers.create_partitions_for_k_fold_validation( bins, folds, boundaries, bin_data, predict_column )
#     print ( len( partition_bin ) )
#     print ( len( partition_bin[0] ) )
#     print ( partition_bin[0][0][0] )
#     print ( partition_bin[1][0][1] )
#     exit()

    total_error_dt = 0
    errors_dt = []
    errors_reg = []
    errors_predict_mean = []
    errors_model = []
    errors_class_pred_mean = []
    tot_min_error_list = []
    tot_model_error_list = []
    tot_no_model_error_list = []
    tot_no_model_min_error_list = []

    predicted_bin_sizes = []

    predictions_df = pd.DataFrame()

    for c in range( folds ):

        X_reg_train, y_reg_train, \
         df_dt_train, y_dt_train, \
          df_dt_test, y_dt_test = helpers.get_train_and_validation_sets_for_fold_c( bins, c, folds, partition_bin )


        # Train decision tree for this fold
        # dt = classifier()
        # dt = classifier( n_estimators = 20 )
        # dt = RandomForestClassifier( n_estimators = 20 )


        # dt = dt.fit( df_dt_train[nominal], y_dt_train )
        # predictions = dt.predict( df_dt_test[nominal] )
        dt = classifire.fit( df_dt_train[attributes_dt], y_dt_train )
        predictions = dt.predict( df_dt_test[attributes_dt] )

#         print ( df_dt_test.shape )
        df_dt_test['Pred_Bins'] = predictions
        dt_predicted_reg_test = df_dt_test.groupby( 'Pred_Bins' )
#         print grp.get_group( 0 )
#         print ( df_dt_test.shape )
#         # print predictions
#         exit()

        prediction_t_or_f = predictions == y_dt_test
        correct = prediction_t_or_f[prediction_t_or_f]
        error = 1 - float( len( correct ) ) / len( predictions )
        # print 'Error Decision Tree fold = {}'.format( error )

        errors_dt.append( error )

        total_error_dt += error

        total_error_reg = 0
        bin_error_reg = []
        bin_size_reg = []
        fold_error_reg = []
        fold_error_predict_mean = []
        fold_error_model = []
        fold_error_class_pred_mean = []
        fold_pred_bin_size = []
        tot_min_err = 0
        tot_min_err_size = 0
        tot_model_error = 0
        tot_no_model_error = 0
        tot_no_model_min_error = 0
        tot_model_error_size = 0

        for b in range( bins ):
            # reg = linear_model.LinearRegression()
            reg = regressor.fit( X_reg_train[b][attributes_reg], X_reg_train[b][predict_column] )
            # reg.fit( X_reg_train[b][attributes_reg], X_reg_train[b][predict_column] )
            reg_models.append( reg )

            # Calculating errors of regression model given correct bin
            pred = reg.predict( partition_bin[b][c][0][attributes_reg] )
            error = abs( pred - partition_bin[b][c][0][predict_column] )
            reg_error = float( sum( error ) ) / len( error )
            fold_error_reg.append( reg_error )
            tot_min_err += sum( error )  # reg_error * len( error )
            tot_min_err_size += len( error )

            # Calculate the bin mean for this bin and this fold
            # bin_mean = partition_bin[b][c][0][predict_column].mean()
            bin_mean = X_reg_train[b][predict_column].mean()

            # Calculating errors
            # prediction: bin bean
            # given correct bin
            no_model_min_errors = abs( bin_mean - partition_bin[b][c][0][predict_column] )
            predict_mean_error = float( sum( no_model_min_errors ) ) / len( no_model_min_errors )
            fold_error_predict_mean.append( predict_mean_error )
            tot_no_model_min_error += sum( no_model_min_errors )

            try:
                # Calculating errors of the final model
                # Bins are found by the classifier and
                # prices are predicted by the regression model
                pred_model = reg.predict( dt_predicted_reg_test.get_group( b )[attributes_reg] )
                error_model = abs( pred_model - dt_predicted_reg_test.get_group( b )[predict_column] )

                # Calculating errors
                # Bins are found by the classifier and
                # prediction bin mean of the training set
                no_model_errors = abs( bin_mean - dt_predicted_reg_test.get_group( b )[predict_column] )

                len_error_model = len( error_model )
                sum_error_model = sum( error_model )
                sum_error_class_pred_mean = sum( no_model_errors )
                sum_no_model_errors = sum( no_model_errors )
                model_error = float( sum_error_model ) / len_error_model
                clas_pred_mean_error = float( sum_error_class_pred_mean ) / len_error_model

                temp_df = dt_predicted_reg_test.get_group( b )
                temp_df['fold_predictions'] = pred_model
                predictions_df = predictions_df.append( temp_df )


            except KeyError:
                len_error_model = 0
                sum_error_model = 0
                sum_no_model_errors = 0
                model_error = 0
                sum_error_class_pred_mean = 0


            fold_error_model.append( model_error )
            fold_error_class_pred_mean.append( clas_pred_mean_error )
            fold_pred_bin_size.append( len_error_model )
            tot_model_error += sum_error_model  # model_error * len( error_model )
            tot_model_error_size += len_error_model
            tot_no_model_error += sum_no_model_errors




#             print ( partition_bin[b][c][1] )
#             print ( pred )
#             exit()


        errors_reg.append( fold_error_reg )
        errors_predict_mean.append( fold_error_predict_mean )
        errors_model.append( fold_error_model )
        errors_class_pred_mean.append( fold_error_class_pred_mean )
        predicted_bin_sizes.append( fold_pred_bin_size )

        tot_min_error_list.append( tot_min_err / tot_min_err_size )
        tot_model_error_list.append( tot_model_error / tot_model_error_size )
        tot_no_model_error_list.append( tot_no_model_error / tot_model_error_size )
        tot_no_model_min_error_list.append( tot_no_model_min_error / tot_model_error_size )


    plt.scatter( predictions_df['SalePrice'], predictions_df['fold_predictions'], s = 80, alpha = 0.4 )
    # plt.title( 'Correlation: {} and r^2: {}'.format( predictions_df['SalePrice'].corr( predictions_df['fold_predictions'] ), r2_score( predictions_df['SalePrice'], predictions_df['fold_predictions'] ) ) )
    plt.axis( [0, max( predictions_df['SalePrice'] ), 0, max( predictions_df['fold_predictions'] )] )
    plt.xlabel( 'Actual Sale Price' )
    plt.ylabel( 'Predicted Sale Price' )
    plt.title( 'Actual Vs. Predicted Sale Price - Correlation {}'.format( predictions_df['SalePrice'].corr( predictions_df['fold_predictions'] ) ) )
    plt.show()

    # bin_matrix_row = [[0] for i in range( bins )]
    # bin_matrix = [bin_matrix_row for i in range( bins )]

    bin_matrix = np.ones( [bins, bins], int )
    # print len( bin_matrix_row ), len( bin_matrix )
    correct_bins = bin_labels.tolist()
    predicted_bins = predictions_df['Pred_Bins'].tolist()
    # print predicted_bins
    for i in range( len( bin_labels ) ):
        # print correct_bins[i], predicted_bins[i]
        bin_matrix[correct_bins[i]][predicted_bins[i]] += 1

    # fig, ax = plt.subplots()
    # heatmap = ax.pcolor( bin_matrix, alpha = 0.8 )
    # plt.imshow( bin_matrix, cmap = 'hot', interpolation = 'nearest' )
    # plt.show()

    # plt.scatter( bin_labels, predictions_df['Pred_Bins'], s = 80, alpha = 0.4 )
    # plt.title( 'Correlation: {} and r^2: {}'.format( predictions_df['SalePrice'].corr( predictions_df['fold_predictions'] ), r2_score( predictions_df['SalePrice'], predictions_df['fold_predictions'] ) ) )
    # plt.show()

    print 'Errors DT:       ', errors_dt
    print
    print 'Tot min error:   ', tot_min_error_list
    print 'Tot model error: ', tot_model_error_list
    print
    print 'Errors Reg:      ', errors_reg
    print 'Errors Model:    ', errors_model
    # print len( errors_reg ), len( errors_reg[0] )
    # print len( predicted_bin_sizes ), len( predicted_bin_sizes[0] )

    bin_tot_min_err = []
    bin_tot_model_err = []
    bin_tot_class_pred_mean_err = []
    bin_tot_min_pred_mean_err = []

    for b in range( bins ):
        tot_min_err = 0
        tot_min_pred_mean_err = 0
        tot_min_err_size = 0
        tot_model_err = 0
        tot_class_pred_mean_err = 0
        tot_model_err_size = 0
        for f in range( folds ):
            tot_min_err += errors_reg[f][b] * len( partition_bin[b][f] )
            tot_min_pred_mean_err += errors_predict_mean[f][b] * len( partition_bin[b][f] )

            tot_min_err_size += len( partition_bin[b][f] )

            tot_model_err += errors_model[f][b] * predicted_bin_sizes[f][b]
            tot_class_pred_mean_err += errors_class_pred_mean[f][b] * predicted_bin_sizes[f][b]
            tot_model_err_size += predicted_bin_sizes[f][b]

        bin_tot_min_err.append( tot_min_err / tot_min_err_size )
        bin_tot_min_pred_mean_err.append( tot_min_pred_mean_err / tot_min_err_size )
        bin_tot_model_err.append( tot_model_err / tot_model_err_size )
        bin_tot_class_pred_mean_err.append( tot_class_pred_mean_err / tot_model_err_size )

        full_bin_tot_min_err.append( bin_tot_min_err )
        full_bin_tot_model_err.append( bin_tot_model_err )
        full_bin_tot_class_pred_mean_err.append( bin_tot_class_pred_mean_err )
        full_bin_tot_min_pred_mean_err.append( bin_tot_min_pred_mean_err )

    print
    print 'Binwise min error:   ', bin_tot_min_err
    print 'Binwise model error: ', bin_tot_model_err

    # Plotting bin wise errors
#     fig3 = plt.figure()
#     ax = fig3.add_subplot( 111 )
#
#     plt.axis( [0, bins, 0, max( bin_tot_min_err + bin_tot_model_err + bin_tot_class_pred_mean_err + bin_tot_min_pred_mean_err ) + 10] )
#
#     line8 = Line2D( range( 1, bins + 1 ), bin_tot_min_err, marker = 'o', markersize = 8, linestyle = '--', label = 'Error ( Regression | Correct Bin )' )
#     ax.add_line( line8 )
#
#     line9 = Line2D( range( 1, bins + 1 ), bin_tot_model_err, color = 'r', marker = '*', markersize = 13, linestyle = '--', label = 'Error ( Classification, Regression )' )
#     ax.add_line( line9 )
#
#     line10 = Line2D( range( 1, bins + 1 ), bin_tot_class_pred_mean_err, color = 'g', marker = 'D', markersize = 8, linestyle = '--', label = 'Error ( Classification, Bin mean )' )
#     ax.add_line( line10 )
#
#     line11 = Line2D( range( 1, bins + 1 ), bin_tot_min_pred_mean_err, color = 'c', marker = 's', markersize = 8, linestyle = '--', label = 'Error ( Bin mean | Correct Bin )' )
#     ax.add_line( line11 )
#
#     plt.legend( handles = [line8, line9, line10, line11], loc = 'upper left' )
#     plt.show()

    print
    print 'Combined dt error:    ', sum( errors_dt ) / len( errors_dt )
    print 'Combined min error:   ', sum( tot_min_error_list ) / len( tot_min_error_list )
    print 'Combined model error: ', sum( tot_model_error_list ) / len( tot_model_error_list )

    decision_tree_bin_errors.append( sum( errors_dt ) / len( errors_dt ) )
    combined_bin_min_errors.append( sum( tot_min_error_list ) / len( tot_min_error_list ) )
    combined_bin_model_errors.append( sum( tot_model_error_list ) / len( tot_model_error_list ) )
    combined_bin_no_model_errors.append( sum( tot_no_model_error_list ) / len( tot_no_model_error_list ) )
    combined_bin_no_model_min_errors.append( sum( tot_no_model_min_error_list ) / len( tot_no_model_min_error_list ) )


print
print decision_tree_bin_errors
print combined_bin_min_errors
print combined_bin_model_errors

min_error = combined_bin_model_errors[0]
ideal_bins = 1
print ideal_bins, min_error
for err in combined_bin_model_errors[1:]:
    if err < min_error:
        min_error = err
        ideal_bins += 1
        print ideal_bins, min_error, combined_bin_model_errors[ideal_bins - 1]
    else:
        break
# ideal_bins = combined_bin_model_errors.index( min( combined_bin_model_errors ) ) + 1
print ideal_bins
# print combined_bin_model_errors[ideal_bins]
# print combined_bin_model_errors[ideal_bins + 1]

fig1 = plt.figure()
ax = fig1.add_subplot( 111 )

x_start = 1

plt.axis( [x_start, len( combined_bin_min_errors ), 0, max( combined_bin_model_errors + combined_bin_no_model_errors )] )

line1 = Line2D( range( x_start, len( combined_bin_no_model_errors ) + 1 ), combined_bin_no_model_errors, color = 'g', marker = 'D', markersize = 8, linestyle = '--', label = 'Error ( Classification, Bin mean )' )
ax.add_line( line1 )

line2 = Line2D( range( x_start, len( combined_bin_model_errors ) + 1 ), combined_bin_model_errors, color = 'r', marker = '*', markersize = 13, linestyle = '--', label = 'Error ( Classification, Regression )' )
ax.add_line( line2 )

#line3 = Line2D( range( x_start, len( combined_bin_no_model_min_errors ) + 1 ), combined_bin_no_model_min_errors, color = 'c', marker = 's', markersize = 8, linestyle = '--', label = 'Error ( Bin mean | Correct Bin )' )
#ax.add_line( line3 )

line4 = Line2D( range( x_start, len( combined_bin_min_errors ) + 1 ), combined_bin_min_errors, marker = 'o', markersize = 8, linestyle = '--', label = 'Error ( Regression | Correct Bin )' )
ax.add_line( line4 )

# plt.xticks( range( 1, bins + 1 ) )

plt.xlabel( 'Number of Bins' )
plt.ylabel( 'Absolute Mean Error' )
plt.title( 'Absolute Mean Error vs. Number of Bins' )
plt.legend( handles = [line1, line2, line4], loc = 'lower left' )

plt.show()

fig2 = plt.figure()
ax = fig2.add_subplot( 111 )

plt.axis( [0, len( decision_tree_bin_errors ), 0, max( decision_tree_bin_errors )] )
line5 = Line2D( range( 1, len( decision_tree_bin_errors ) + 1 ), decision_tree_bin_errors, marker = 'o', markersize = 8, linestyle = '--', label = 'Error ( Classification )' )
ax.add_line( line5 )

plt.xlabel( 'Number of Bins' )
plt.ylabel( 'Error Ratio' )
plt.title( 'Classification Error vs. Number of Bins' )
plt.legend( handles = [line5], loc = 'upper left' )

plt.show()

# Train the final model using
# ideal bins and the full new_train set
bins, bin_data, boundaries = helpers.create_equal_frequency_bins( samples, ideal_bins, new_train, sorted_hp, predict_column )
bin_labels = pd.Series( [helpers.sale_price_to_bin( boundaries, sp ) for sp in new_train[predict_column]] )
dt = classifire.fit( new_train[attributes_dt], bin_labels )

for b in range( ideal_bins ):
    print 'training {}'.format( b )
    # reg = linear_model.LinearRegression()
    # reg.fit( bin_data[b][attributes_reg], bin_data[b][predict_column] )
    reg = regressor.fit( bin_data[b][attributes_reg], bin_data[b][predict_column] )
    reg_models.append( reg )

# Test the final model on the new_test set
test = new_test
pred = dt.predict( test[attributes_dt] )
test['prediction'] = pred
test_bins = test.groupby( 'prediction' )

print 'test shape ', test.shape

predictions_df = pd.DataFrame()
for b in range( ideal_bins ):
    print 'predicting {}'.format( b )
    try:

        reg_pred = reg_models[b].predict( test_bins.get_group( b )[attributes_reg] )
        print b, len( reg_pred ), type( reg_pred )

        temp_df = test_bins.get_group( b )
        temp_df[predict_column] = reg_pred

        predictions_df = predictions_df.append( temp_df )

    except:
        print 'Except'
        continue

print predictions_df.shape

predictions_df.sort_index( axis = 0 )
# print ( predictions_df[predict_column] )
predictions_df[predict_column].to_csv( '++Test_set_predictions.csv' )


# plt.scatter( predictions_df['SalePrice'], predictions_df['predict_column'], s = 80, alpha = 0.4 )
# plt.show()
exit()

for grp in test_bins.groups:

    # Number of examples in this group
    g_size = len( test_bins.get_group( grp ) )

    print test_bins.get_group( grp )
