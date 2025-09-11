import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 

#------------------------------------------------------------------------------------
# Fit vào decision tree và chọn ra K feature tốt nhất
#------------------------------------------------------------------------------------


def Feature_Selection(self):
    print("START SELECTING DATA..." + '\n' + '=' * 100)

    #get correlation rank
    print('Fitting into Decision Tree...')
    model_ = DecisionTreeClassifier(random_state = 42)
    model_.fit(self.X_train, self.y_train)

    importances = pd.Series(model_.feature_importances_, index = self.X_train.columns)
    importances_sorted = importances.sort_values(ascending=False)
    print("Completed" + '\n' + '=' * 100)

    #get top K
    print('Getting top K features...')
    col_4 = importances_sorted.head(4).index.tolist()
    col_8 = importances_sorted.head(8).index.tolist() 
    col_16 = importances_sorted.head(16).index.tolist()
    col_20 = importances_sorted.head(20).index.tolist()

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
    return (importances_sorted, 
            self.X_train_selection_4, 
            self.X_test_selection_4, 
            self.X_train_selection_8, 
            self.X_test_selection_8, 
            self.X_train_selection_16, 
            self.X_test_selection_16,
            self.X_train_selection_20,
            self.X_test_selection_20)

