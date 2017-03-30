'''
Created on Mar 5, 2017

@author: Manujinda Wathugala
Binning the data into different number of bins
and training regression models and checking their accuracy.
'''
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from multi_column_label_encoder import MultiColumnLabelEncoder
import pandas as pd


def sale_price_to_bin( boundaries, sale_price ):
    b = 0
    while boundaries[b] <= sale_price:
        b += 1

    return b - 1

# boun = [0, 10, 20, 30]
# print sale_price_to_bin( boun, 20 )
# exit()
converters = {
              'MasVnrType': lambda x: None if x == 'NA' else x,
              'Electrical': lambda x: None if x == 'NA' else x,
              'LotFrontage': lambda x: None if x == 'NA' else int( x ),
              'MasVnrArea': lambda x: 0 if x == 'NA' else int( x ),
              'GarageYrBlt': lambda x: 0 if x == 'NA' else int( x ),
              }

numeric = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
nominal = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

house_data = pd.read_csv( 'train.csv', index_col = 0, na_filter = False, converters = converters )

test_data = pd.read_csv( 'test.csv', index_col = 0, na_filter = False, converters = converters )

# house_data = pd.read_csv( 'small.csv' )
# print house_data.head()
# 'LotFrontage': lambda x: 0 if x == 'NA' else int( x ),
# 'MasVnrArea': lambda x:-100 if x == 'NA' else int( x ),

# Filling missing values
house_data[['MasVnrArea']] = house_data[['MasVnrArea']].fillna( house_data[['MasVnrArea']].mean() )
house_data[['LotFrontage']] = house_data[['LotFrontage']].fillna( house_data[['LotFrontage']].mean() )
house_data[['MasVnrType']] = house_data[['MasVnrType']].fillna( house_data[['MasVnrType']].mode() )
house_data[['Electrical']] = house_data[['Electrical']].fillna( house_data[['Electrical']].mode() )

test_data[['MasVnrArea']] = test_data[['MasVnrArea']].fillna( house_data[['MasVnrArea']].mean() )
test_data[['LotFrontage']] = test_data[['LotFrontage']].fillna( house_data[['LotFrontage']].mean() )
test_data[['MasVnrType']] = test_data[['MasVnrType']].fillna( house_data[['MasVnrType']].mode() )
test_data[['Electrical']] = test_data[['Electrical']].fillna( house_data[['Electrical']].mode() )

# print ( house_data )
# house_data[['MasVnrArea']] = house_data[['MasVnrArea']].fillna( -10000 )
# print house_data[['MasVnrArea']].mean()
print test_data.shape
zero_price = [0 for i in range( test_data.shape[0] )]
test_data['SalePrice'] = zero_price
print test_data.shape

combined = pd.concat( [house_data, test_data] )

# print ( combined.shape )

# print ( combined.tail() )
dummies = pd.get_dummies( combined, columns = nominal )
train_data_w_sale = dummies[ :house_data.shape[0]]
test_data = dummies[house_data.shape[0]:]
#print(dummies.head())

#delete sale price to use in training
train_data = train_data_w_sale.drop('SalePrice',1)
# print ( dummies.head() )
# print ( dummies.shape )

#changed = MultiColumnLabelEncoder( columns = nominal ).fit_transform( combined )



#new_train = changed[:][ :house_data.shape[0]]
#new_test = changed[:][house_data.shape[0]:]

#print new_train.shape
#print new_test.shape

# new_train.to_csv( 'new_train.csv' )
# new_test.to_csv( 'new_test.csv' )




sorted_hp = sorted( house_data['SalePrice'] )
samples = house_data.shape[0]

# Try different numbers of bins
for bins in range( 2, 3 ):
    # freq = samples / bins
    # print 'Frequencey {}'.format( freq )
    # rem = samples % bins
    # print rem

    # lo = sorted_hp[0]
    bin_data = []

    # p_index = 0

    lo_index = 0
    hi_index = 0

    print ( '\nBins {}'.format( bins ) )
    tot = 0
    boundaries = [0]
    last_bin = False

    # Bin samples in to bins
    for b in range( bins ):

        # Remaining samples yet to be binned
        remaining_samples = samples - tot
        # Remaining number of bins
        remaining_bins = bins - b

        # Frequency for the remaining bins.
        freq = int( remaining_samples / remaining_bins )
        # Excess samples that needs to be evenly distributed
        # among the bins. The pigeon hole principle.
        excess_samples = remaining_samples % remaining_bins

        # If there are excess samples, increase the bin frequency
        # if b < rem:
        if excess_samples != 0:
            hi_index += freq + 1
        else:
            hi_index += freq

        if hi_index >= samples:
            # This is the last bin.
            # Put all the remaining examples in this bin
            hi_index = samples - 1

            # Since this is the last bin we need to include
            # samples with target value sorted_hp[hi_index]
            # in this bin.
            # Since our criteria to select samples for a bin is
            # 'SalePrice < hi, add 1 to sorted_hp[hi_index]
            # so that we can include these sample in the last bin
            hi = sorted_hp[hi_index] + 1
            last_bin = True
        else:
            # Borderline value for this bin.
            # hi will fall in to the next bin
            hi = sorted_hp[hi_index]

            # We do not want samples with the same target value
            # to be in two bins.

            # Count the number of samples with indexes >= hi_index
            # with the same hi value
            up = 1
            while hi_index + up < samples and sorted_hp[hi_index + up] == hi:
                up += 1

            # Count the number of samples with indexes < hi_index
            # with the same hi value
            down = 1
            while hi_index - down >= 0 and sorted_hp[hi_index - down] == hi:
                down += 1
            down -= 1

            # Decide whether to put all the samples with the target
            # value hi in this bin or the next bin.
            if down > up:
                # We already have more samples with the target value hi
                # in this bin. So include all the samples with target value
                # hi in this bin
                hi_index += up
            else:
                # More samples with the target value hi falls in the next bin
                if hi_index - down - 1 <= lo_index:
                    # This bin will be empty. So include all the samples with
                    # target value hi in this bin.
                    hi_index += up
                else:
                    # Include all the samples with target value hi
                    # in the next bin.
                    hi_index -= down

            # After deciding which bin the multiple copies
            # go into, check whether we are at the last bin.
            if hi_index >= samples:
                hi_index = samples - 1
                hi = sorted_hp[hi_index] + 1
                last_bin = True
            else:
                # Borderline value for this bin after adjusting for
                # sample with the same target value.
                # hi will fall in to the next bin
                hi = sorted_hp[hi_index]


#         hi = sorted_hp[hi_index]
#         hi = hi + 1 if last_bin else hi
        lo = sorted_hp[lo_index]
        print ( lo_index, hi_index )

        temp_df = house_data[( house_data.SalePrice >= lo ) & ( house_data.SalePrice < hi )]
        temp_df_reset = temp_df.reset_index( drop = True )
        bin_data.append( temp_df_reset )

        boundaries.append( hi )
        lo_index = hi_index

        print ( bin_data[b].shape )
        tot += bin_data[b].shape[0]

        if last_bin:
            bins = b + 1
            break

    print ( 'Total {}'.format( tot ) )
    print ( bins )
    print ( boundaries )

    bin_labels = pd.Series( [sale_price_to_bin( boundaries, sp ) for sp in house_data['SalePrice']] )

    # print ( bin_labels[:10] )

    house_data['bin'] = bin_labels
    # print ( house_data[['SalePrice', 'bin']][:10] )

    folds = 5
    total_error = 0

    kf = KFold( n_splits = folds )
    kf.get_n_splits( train_data_w_sale )

    for train_index, test_index in kf.split( train_data_w_sale ):

        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_test = bin_labels[train_index], bin_labels[test_index]

        #dt = tree.DecisionTreeClassifier()
        #dt = dt.fit( X_train, y_train )
        #predictions = dt.predict( X_test )
        clf = SVC(kernel='linear')
        clf = clf.fit(X_train, y_train)
        predictions = clf.predict( X_test )

        prediction_t_or_f = predictions == y_test
        correct = prediction_t_or_f[prediction_t_or_f]
        error = 1 - float( len( correct ) ) / len( predictions )
        print 'Error SVM Kernel fold = {}'.format( error )

        total_error += error

    total_error /= folds
    print 'Error SVM Kernel = {}'.format( total_error )


    clf = clf.fit( train_data, bin_labels )
    predictions = clf.predict(train_data )
    prediction_t_or_f = predictions == bin_labels
    correct = prediction_t_or_f[prediction_t_or_f]
    error = 1 - float( len( correct ) ) / len( predictions )
    print 'Error = {}'.format( error )
    # zipped = zip( predictions, bin_labels )
    # print zipped[:20]

#     for attrib in nominal:
#         dt = tree.DecisionTreeClassifier()
#         dt = dt.fit( new_train[[attrib]], bin_labels )
#         predictions = dt.predict( new_train[[attrib]] )
#
#
#         prediction_t_or_f = predictions == bin_labels
#         correct = prediction_t_or_f[prediction_t_or_f]
#         accuracy = float( len( correct ) ) / len( predictions )
#         print 'Accuracy {} = {}'.format( attrib, accuracy )

    bin_size = []
    bin_error = []
    for b in range( bins ):
        if bin_data[b].shape[0] != 0:

            folds = 5
            total_error = 0

            kf = KFold( n_splits = folds )
            kf.get_n_splits( bin_data[b] )

            for train_index, test_index in kf.split( bin_data[b] ):
                # print ( train_index )
                # print ( test_index )
                # continue

                # print ( bin_data[b].iloc[train_index][numeric] )
                X_train, X_test = bin_data[b].iloc[train_index][numeric], bin_data[b].iloc[test_index][numeric]
                y_train, y_test = bin_data[b].iloc[train_index]['SalePrice'], bin_data[b].iloc[test_index]['SalePrice']

                #reg = linear_model.LinearRegression()
                ada = AdaBoostRegressor()
                ada=ada.fit( X_train, y_train )
                # reg.fit( bin_data[b][numeric], bin_data[b]['SalePrice'] )
                #reg.fit( X_train, y_train )

                # pred = reg.predict( bin_data[b][numeric] )
                #pred = reg.predict( X_test )
                pred = ada.predict( X_test )

                # error = abs( pred - bin_data[b]['SalePrice'] )
                error = abs( pred - y_test )

                # error_sq = error * error
                # model_error = sum( error_sq ) / len( error_sq )
                model_error = sum( error ) / len( error )

                total_error += model_error


            total_error /= folds
            bin_error.append( total_error )
            bin_size.append( bin_data[b].shape[0] )

            print ( 'Absolute Mean Error Model: ' ),
            print ( model_error )

    total_error = 0
    print ( bin_size )
    print ( bin_error )
    for b in range( len( bin_size ) ):
        total_error += bin_size[b] * bin_error[b]

    total_error /= sum( bin_size )
    print ( 'Total Absolute Mean Error Model: ' ),
    print ( total_error )


exit()






# print house_data[['MasVnrArea', 'LotFrontage']]
# print house_data[numeric]

# reg = linear_model.Lasso( alpha = 0.1 )
# reg = linear_model.LassoLars( alpha = .1 )
# reg = linear_model.Ridge ( alpha = .5 )
reg = linear_model.LinearRegression()
# reg = svm.SVR( kernel = 'rbf' )

reg.fit( house_data[numeric], house_data['SalePrice'] )

pred = reg.predict( house_data[numeric] )

error = pred - house_data['SalePrice']

error_sq = error * error

# print( metrics.classification_report( house_data['SalePrice'], pred ) )

# print pred[:10]
# print house_data['SalePrice'][:10]
# print error[:10]
# print error_sq[:10]

model_error = sum( error_sq ) / len( error_sq )
print ( 'Squared Error Model: ', )
print ( model_error )


mean_price = house_data['SalePrice'].mean()
# print mean_price
mean_error = mean_price - house_data['SalePrice']
mean_err_sq = mean_error * mean_error
mean_error = sum( mean_err_sq ) / len( error_sq )
print ( 'Squared Error if the prediction is always the mean: ', )
print ( mean_error )
# print house_data['SalePrice'][:5]
# print mean_error[:5]

print ( 'Mean error - model error (Larger the better): ', )
print ( mean_error - model_error )

'''
X = [[1], [2], [3], [4]]
y = [1, 2, 3, 4]
reg = linear_model.LinearRegression()
reg.fit( X, y )
pred = reg.predict( X )
error = pred - y
error_sq = error * error
print pred
print error_sq
'''
