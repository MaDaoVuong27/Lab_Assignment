import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def Feature_Extraction(self):
    print("START EXTRACTING DATA..." + '\n' + '=' * 100)
    original_feature = self.X_train_scaled.shape[1]
    self.PCA_4 = PCA(n_components=4, random_state=self.random_state)
    self.PCA_8 = PCA(n_components=8, random_state=self.random_state)
    self.PCA_16 = PCA(n_components=16, random_state=self.random_state)
    self.PCA_20 = PCA(n_components=20, random_state=self.random_state)

    #4
    print("Extracting to 4 features...")
    X_train_extraction_4 = self.X_train_scaled.copy()
    X_test_extraction_4 = self.X_test_scaled.copy()

    X_train_extraction_4 = pd.DataFrame(self.PCA_4.fit_transform(X_train_extraction_4))
    X_test_extraction_4 = pd.DataFrame(self.PCA_4.transform(X_test_extraction_4))

    print(f'Extracted {original_feature} features to {X_train_extraction_4.shape[1]} features')
    print("Completed" + '\n' + '=' * 100)

    #8
    print("Extracting to 8 features...")
    X_train_extraction_8 = self.X_train_scaled.copy()
    X_test_extraction_8 = self.X_test_scaled.copy()
 
    X_train_extraction_8 = pd.DataFrame(self.PCA_8.fit_transform(X_train_extraction_8))
    X_test_extraction_8 = pd.DataFrame(self.PCA_8.transform(X_test_extraction_8))

    print(f'Extracted {original_feature} features to {X_train_extraction_8.shape[1]} features')
    print("Completed" + '\n' + '=' * 100)

    #16
    print("Extracting to 16 features...")
    X_train_extraction_16 = self.X_train_scaled.copy()
    X_test_extraction_16 = self.X_test_scaled.copy()

    X_train_extraction_16 = pd.DataFrame(self.PCA_16.fit_transform(X_train_extraction_16))
    X_test_extraction_16 = pd.DataFrame(self.PCA_16.transform(X_test_extraction_16))

    print(f'Extracted {original_feature} features to {X_train_extraction_16.shape[1]} features')
    print("Completed" + '\n' + '=' * 100)

    #20
    print("Extracting to 20 features...")
    X_train_extraction_20 = self.X_train_scaled.copy()
    X_test_extraction_20= self.X_test_scaled.copy()

    X_train_extraction_20 = pd.DataFrame(self.PCA_20.fit_transform(X_train_extraction_20))
    X_test_extraction_20 = pd.DataFrame(self.PCA_20.transform(X_test_extraction_20))

    print(f'Extracted {original_feature} features to {X_train_extraction_20.shape[1]} features')
    print("Completed" + '\n' + '=' * 100)

    #return
    self.X_train_extraction_4 = X_train_extraction_4
    self.X_test_extraction_4 = X_test_extraction_4
    self.X_train_extraction_8 = X_train_extraction_8
    self.X_test_extraction_8 = X_test_extraction_8
    self.X_train_extraction_16 = X_train_extraction_16
    self.X_test_extraction_16 = X_test_extraction_16
    self.X_train_extraction_20 = X_train_extraction_20
    self.X_test_extraction_20 = X_test_extraction_20

    print('EXTRACTING DATA SUCCESSFULLY!')
    return (self.X_train_extraction_4, 
            self.X_test_extraction_4, 
            self.X_train_extraction_8, 
            self.X_test_extraction_8, 
            self.X_train_extraction_16, 
            self.X_test_extraction_16, 
            self.X_train_extraction_20,
            self.X_test_extraction_20)
    