import pandas as pd
import numpy as np

#------------------------------------------------------------------------------------
# Tính corr matrix giữa các feature, không dùng abs()
# Tính trung bình, sắp xếp giảm dần, lấy K feature cao nhất
#------------------------------------------------------------------------------------


def Feature_Selection(self):
    print("START SELECTING DATA..." + '\n' + '=' * 100)

    #get correlation rank
    print('Calculating correlation matrix...')
    features = self.train_processed.select_dtypes(include = [np.number])
    features = features.drop(['label'], axis = 1)
    
    corr_matrix = features.corr()
    np.fill_diagonal(corr_matrix.values, np.nan)
    avr_corr = corr_matrix.mean(skipna = True)
    avr_corr_sorted = avr_corr.sort_values(ascending = False)
    print("Completed" + '\n' + '=' * 100)

    #get top K
    print('Getting top K features...')
    col_4 = avr_corr_sorted.head(4).index.tolist()
    col_8 = avr_corr_sorted.head(8).index.tolist()
    col_16 = avr_corr_sorted.head(16).index.tolist()
    col_20 = avr_corr_sorted.head(20).index.tolist()

    self.X_train_selection_4 = self.X_train[col_4].copy()
    self.X_test_selection_4 = self.X_test[col_4].copy()
    self.X_train_selection_8 = self.X_train[col_8].copy()
    self.X_test_selection_8 = self.X_test[col_8].copy()
    self.X_train_selection_16 = self.X_train[col_16].copy()
    self.X_test_selection_16 = self.X_test[col_16].copy()
    self.X_train_selection_20 = self.X_train[col_20].copy()
    self.X_test_selection_20 = self.X_test[col_20].copy()
    print("Completed" + '\n' + '=' * 100)

    #retutn
    print('SELECTING DATA SUCCESSFULLY!')
    return (avr_corr_sorted, 
            self.X_train_selection_4, 
            self.X_test_selection_4, 
            self.X_train_selection_8, 
            self.X_test_selection_8, 
            self.X_train_selection_16, 
            self.X_test_selection_16,
            self.X_train_selection_20,
            self.X_test_selection_20)

