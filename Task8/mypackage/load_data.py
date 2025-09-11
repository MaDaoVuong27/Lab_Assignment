#from . import pd, MinMaxScaler, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, MLPClassifier, BernoulliNB
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier


class UNSW_NB15:
    def __init__(self):
        self.random_state = 42
        self.scaler = MinMaxScaler()
        self.onehot_encoder = {}
        self.label_encoder = {}
        self.corr = pd.Series()
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state = self.random_state), 
            'Random Forest': RandomForestClassifier(random_state = self.random_state, max_depth = 5),
            'K Neighbors': KNeighborsClassifier(n_neighbors = 5),
            'MLP': MLPClassifier(random_state = self.random_state, max_iter = 100, hidden_layer_sizes = (200,)),
            'Naive Bayes': BernoulliNB()
        }
        self.train = None
        self.test = None

    def load_data(self, train_path, test_path):
        #load data
        print("LOADING UNSW_NB15 DATASET...")

        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

        print("LOADING UNSW_NB15 DATASET SUCCESSFULLY!") 

        #overview
        print('=' * 100)
        print("Train and Test data overview")

        print(f"Training data shape: {self.train.shape}")
        print(f"Training data columns: {list(self.train.columns)}")
        print(f"Training data info\n: {self.train.info}")
        print(f"Training data check null:\n {self.train.isnull().sum()}")
        print(f"Training data check duplicate: {self.train.duplicated().sum()}")
        print(f"Training data first 5 columns:\n {self.train.head(5)}")


        print(f"Test data shape: {self.test.shape}")
        print(f"Test data columns: {list(self.test.columns)}")
        print(f"Test data info\n: {self.test.info}")
        print(f"Test data check null:\n {self.test.isnull().sum()}")
        print(f"Test data check duplicate: {self.test.duplicated().sum()}")
        print(f"Test data first 5 columns:\n {self.test.head(5)}")

        #Class distribution
        print('=' * 100)
        print("Class distribution")

        print("Class distribution in Training data ('label'):")
        print(self.train['label'].value_counts())

        print("\nAttack categories distribution in Training Data ('attack_cat'):")
        print(self.train['attack_cat'].value_counts())

        print("\nClass distribution in Testing Data ('label'):")
        print(self.test['label'].value_counts())

        print("\nAttack categories distribution in Testing Data ('attack_cat'):")
        print(self.test['attack_cat'].value_counts())


        #Correlation
        self.corr = (self.train.corr(numeric_only=True)['label'].drop(['label', 'id']).abs().sort_values(ascending=False))    
        print("\nCorrelation between feature and label:")
        print(self.corr)

        #return
        return self.train, self.test
