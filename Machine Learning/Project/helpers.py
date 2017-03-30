'''
Created on Mar 17, 2017

@author: manujinda
'''
from sklearn.model_selection import train_test_split

import pandas as pd


def sale_price_to_bin( boundaries, sale_price ):
    b = 0
    while boundaries[b] <= sale_price:
        b += 1

    return b - 1


def create_equal_frequency_bins( samples, bins, data_frame, sorted_hp, predict_column ):
    bin_data = []

    lo_index = 0
    hi_index = 0

    print ( '\nTotal bins to create : {}'.format( bins ) )
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
        # print ( lo_index, hi_index )

        temp_df = data_frame[( data_frame[predict_column] >= lo ) & ( data_frame[predict_column] < hi )]
        temp_df_reset = temp_df.reset_index( drop = True )
        bin_data.append( temp_df_reset )

        boundaries.append( hi )
        lo_index = hi_index

        # print ( bin_data[b].shape )
        tot += bin_data[b].shape[0]

        # Create bin histograms
#         print '\t\t### Hist bin {}'.format( b )
#         plt.hist( bin_data[b]['SalePrice'], 10 )
#
#         plt.title( 'Sale Price Distribution - {} Bins - Bin {}'.format( bins, b ) )
#         plt.ylabel( 'Number of houses' )
#         plt.xlabel( 'Sale Prices' );
#
#         plt.savefig( os.path.join( plot_dir, '{}_Bins_Bin_{}.png'.format( bins, b ) ) )
#         # plt.show()
#         plt.close()
#         plt.cla()
#         plt.clf()

        if last_bin:
            bins = b + 1
            break


    return bins, bin_data, boundaries



def create_partitions_for_k_fold_validation( bins, folds, boundaries, bin_data, predict_column ):
    partition_bin = []
    for b in range( bins ):
        bin_part = []
        X_train = bin_data[b]
        y_train = pd.Series( [sale_price_to_bin( boundaries, sp ) for sp in X_train[predict_column]] )

        for f in range( folds, 1, -1 ):
            percent_test = 1.0 / f
            X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = percent_test )
            bin_part.append( ( X_test, y_test ) )

        bin_part.append( ( X_train, y_train ) )

        partition_bin.append( bin_part )

    return partition_bin


def get_train_and_validation_sets_for_fold_c( bins, c, folds, partition_bin ):
    X_reg_train = []
    y_reg_train = []

    for b in range( bins ):
        df_train = pd.DataFrame()
        y_train = pd.Series()

        for c2 in range( folds ):
            if c2 != c:
                df_train = df_train.append( partition_bin[b][c2][0] )
                y_train = y_train.append( partition_bin[b][c2][1] )

        X_reg_train.append( df_train )
        y_reg_train.append( y_train )

    df_dt_train = pd.DataFrame()
    y_dt_train = pd.Series()

    df_dt_test = pd.DataFrame()
    y_dt_test = pd.Series()

    # Create training and validation sets for decision tree
    for b in range( bins ):
        df_dt_train = df_dt_train.append( X_reg_train[b] )
        y_dt_train = y_dt_train.append( y_reg_train[b] )

        df_dt_test = df_dt_test.append( partition_bin[b][c][0] )
        y_dt_test = y_dt_test.append( partition_bin[b][c][1] )

#     print '--------------'
#     print ( df_dt_train.shape )
#     print ( len( y_dt_train ) )
#     print ( df_dt_test.shape )
#     print ( len( y_dt_test ) )

    return X_reg_train, y_reg_train, df_dt_train, y_dt_train, df_dt_test, y_dt_test


