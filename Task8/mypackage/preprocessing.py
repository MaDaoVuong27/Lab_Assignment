#from . import pd, np, OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder

def Preprocess(self):
    print("START PREPROCESSING DATA..." + '\n' + '=' * 100)

    #rip 'id'
    print("Remove unnecessary columns...")
    to_drop = ['id']
    self.train_processed = self.train.drop(to_drop, axis = 1)
    self.test_processed = self.test.drop(to_drop, axis = 1)
    print("Completed" + '\n' + '=' * 100)
    """
    #Replace inf value = NaN
    print("Handling -inf and inf values in Train and Test...")
    self.train_processed = self.train_processed.replace([np.inf, -np.inf], np.nan)
    self.test_processed = self.test_processed.replace([np.inf, -np.inf], np.nan)
    print("Completed" + '\n' + '=' * 100)

    #Drop NaN row
    print("Dropping rows that contain NaN value in Train and Test...")
    row_before = len(self.train_processed)
    self.train_processed = self.train_processed.dropna()
    row_after = len(self.train_processed)
    print(f'Train: Dropped {row_before - row_after}, from {row_before} to {row_after}\n')

    row_before = len(self.test_processed)
    self.test_processed = self.test_processed.dropna()
    row_after = len(self.test_processed)
    print(f'Test: Dropped {row_before - row_after}, from {row_before} to {row_after}\n')
    print("Completed" + '\n' + '=' * 100)
    """
    #split x, y
    print("Separate feature and target...")
    features = [col for col in self.train_processed.columns if col not in ['attack_cat', 'label']]

    X_train = self.train_processed[features].copy()
    X_test = self.test_processed[features].copy()
    y_train_binary = self.train_processed['label'].copy()
    y_test_binary = self.test_processed['label'].copy()
    y_train_multi = self.train_processed['attack_cat'].copy()
    y_test_multi = self.test_processed['attack_cat'].copy()
    print("Completed" + '\n' + '=' * 100)

    #Indentify numerical and categorical
    print("Indentify numerical and categorical columns (no target columns)...")
    numerical_col = X_train.select_dtypes(include = [np.number]).columns
    categorical_cols = X_train.select_dtypes(include = [object]).columns

    print(f'Total features: {len(features)}')
    print(f'Total numerical features: {len(numerical_col)}')
    print(f'Total categorical features: {len(categorical_cols)}')
    print("Completed" + '\n' + '=' * 100)

    #onehotencoding
    print('Applying Onehot enconding for categorical features...')
    print(f'X_train shape before Onehot encoding: {X_train.shape}')
    for col in categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')
        #Fit on combined data to ensure consistent encoding
        #combined_col = pd.concat([X_train[col], X_test[col]], ignore_index=True).to_frame()
        #ohe.fit(combined_col.astype(str))
        ohe.fit(X_train[[col]].astype(str))
        ohe_cols = ohe.get_feature_names_out()

        train_onehot = ohe.transform(X_train[[col]].astype(str))
        test_onehot = ohe.transform(X_test[[col]].astype(str))

        train_onehot_df = pd.DataFrame(train_onehot, columns = ohe_cols, index = X_train.index)
        test_onehot_df =  pd.DataFrame(test_onehot, columns = ohe_cols, index = X_test.index)

        X_train = pd.concat([X_train.drop(columns = col), train_onehot_df], axis = 1)
        X_test = pd.concat([X_test.drop(columns = col), test_onehot_df], axis = 1)

        self.onehot_encoder[col] = ohe

    print(f'X_train shape after Onehot encoding: {X_train.shape}')
    print("Completed" + '\n' + '=' * 100)

    #Scale
    print('Applying MinMaxScaler for numerical features...')
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_col] = self.scaler.fit_transform(X_train_scaled[numerical_col])
    X_test_scaled[numerical_col] = self.scaler.transform(X_test_scaled[numerical_col])
    print("Completed" + '\n' + '=' * 100)

    #Label encoding
    print('Applying Label enconding for target features...')
    self.multi_label_encoder = LabelEncoder()

    combined_multi = pd.concat([y_train_multi, y_test_multi], ignore_index=True)
    self.multi_label_encoder.fit(combined_multi.astype(str))
    self.multi_classes_ = self.multi_label_encoder.classes_.tolist()     

    y_train_multi_encoded = self.multi_label_encoder.transform(y_train_multi.astype(str))
    y_test_multi_encoded = self.multi_label_encoder.transform(y_test_multi.astype(str))
    
    print("Completed" + '\n' + '=' * 100)

    #return
    self.X_train = X_train
    self.X_test = X_test
    self.X_train_scaled = X_train_scaled
    self.X_test_scaled = X_test_scaled
    self.y_train_binary = y_train_binary
    self.y_test_binary = y_test_binary
    self.y_train_multi = y_train_multi_encoded
    self.y_test_multi = y_test_multi_encoded

    print('PREPROCESSING DATA SUCCESSFULLY!') 
    return self.X_train, self.X_test, self.X_train_scaled, self.X_test_scaled, self.y_train_binary, self.y_test_binary, self.y_train_multi, self.y_test_multi


# Add method to UNSW_NB15 class
#from load_data import UNSW_NB15
#UNSW_NB15.preprocess = Preprocess
