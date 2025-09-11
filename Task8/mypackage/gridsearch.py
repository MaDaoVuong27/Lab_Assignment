import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score



def Grid_Search_Decision_Tree(x_train_, y_train_, x_test_, y_test_):
    model = DecisionTreeClassifier(random_state = 42)

    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],   # Hàm đo độ phân chia
        "max_depth": [None, 5, 10, 20],             # Độ sâu tối đa của cây
        "min_samples_split": [2, 5, 10],            # Số mẫu tối thiểu để tách nhánh
        "min_samples_leaf": [1, 2, 5],              # Số mẫu tối thiểu ở một lá
    }

    grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'f1_weighted', cv = 4, verbose = 2)
    grid.fit(x_train_, y_train_)
    y_predict_ = grid.predict(x_test_)
    precision = precision_score(y_test_, y_predict_, average = 'weighted')
    recall = recall_score(y_test_, y_predict_, average = 'weighted')
    f1 = f1_score(y_test_, y_predict_, average = 'weighted')

    print(f'Precision: {precision * 100:.2f}')
    print(f'Recall: {recall * 100:.2f}')
    print(f'F1-score: {f1 * 100:.2f}')

    best_model = grid.best_estimator_
    print(f'BEST MODEL: {best_model}')
    print(f'BEST SCORE: {grid.best_score_ * 100:.2f}')
    print(f'BEST PARAMS: {grid.best_params_}')

def Grid_Search_Random_Forest(x_train_, y_train_, x_test_, y_test_):
    model = RandomForestClassifier(random_state = 42, n_jobs = -1)

    param_grid = {
        "n_estimators": [100, 200, 300],            # số lượng cây trong rừng
        "criterion": ["gini", "entropy", "log_loss"], # hàm đánh giá độ phân chia
        "max_depth": [None, 10, 20],            # độ sâu tối đa của cây
        "min_samples_split": [2, 5, 7],            # số mẫu tối thiểu để tách nhánh
        "min_samples_leaf": [1, 2, 3],              # số mẫu tối thiểu ở một lá
    }

    grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'f1_weighted', cv = 4, verbose = 2)
    grid.fit(x_train_, y_train_)
    y_predict_ = grid.predict(x_test_)
    precision = precision_score(y_test_, y_predict_, average = 'weighted')
    recall = recall_score(y_test_, y_predict_, average = 'weighted')
    f1 = f1_score(y_test_, y_predict_, average = 'weighted')

    print(f'Precision: {precision * 100:.2f}')
    print(f'Recall: {recall * 100:.2f}')
    print(f'F1-score: {f1 * 100:.2f}')

    best_model = grid.best_estimator_
    print(f'BEST MODEL: {best_model}')
    print(f'BEST SCORE: {grid.best_score_ * 100:.2f}')
    print(f'BEST PARAMS: {grid.best_params_}')

def Grid_Search_K_Neighbors(x_train_, y_train_, x_test_, y_test_):  
    model = KNeighborsClassifier(n_jobs = -1)

    param_grid = {
        "n_neighbors": [3, 5, 7, 9],           # số lượng láng giềng
        "weights": ["uniform", "distance"],        # cách tính trọng số
        "metric": ["euclidean", "manhattan", "minkowski"], # hàm khoảng cách
        "p": [1, 2]                                # chỉ số p cho Minkowski (1=Manhattan, 2=Euclidean)
    }

    grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'f1_weighted', cv = 4, verbose = 2)
    grid.fit(x_train_, y_train_)
    y_predict_ = grid.predict(x_test_)
    precision = precision_score(y_test_, y_predict_, average = 'weighted')
    recall = recall_score(y_test_, y_predict_, average = 'weighted')
    f1 = f1_score(y_test_, y_predict_, average = 'weighted')

    print(f'Precision: {precision * 100:.2f}')
    print(f'Recall: {recall * 100:.2f}')
    print(f'F1-score: {f1 * 100:.2f}')

    best_model = grid.best_estimator_
    print(f'BEST MODEL: {best_model}')
    print(f'BEST SCORE: {grid.best_score_ * 100:.2f}')
    print(f'BEST PARAMS: {grid.best_params_}')

def Grid_Search_MLP(x_train_, y_train_, x_test_, y_test_):
    model = MLPClassifier(random_state = 42)

    param_grid = {
        "hidden_layer_sizes": [(100,), (100, 50), (50, 50, 50)],  # cấu trúc hidden layers
        "activation": ["relu", "tanh"],  # hàm kích hoạt
        "solver": ["adam", "sgd"],          # thuật toán tối ưu
        "alpha": [0.0001, 0.001],              # hệ số regularization (L2 penalty)
        "learning_rate": ["constant", "adaptive"],   # cách thay đổi learning rate
        "max_iter": [200, 300]                       # số vòng lặp tối đa
    }

    grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'f1_weighted', cv = 4, verbose = 2)
    grid.fit(x_train_, y_train_)
    y_predict_ = grid.predict(x_test_)
    precision = precision_score(y_test_, y_predict_, average = 'weighted')
    recall = recall_score(y_test_, y_predict_, average = 'weighted')
    f1 = f1_score(y_test_, y_predict_, average = 'weighted')

    print(f'Precision: {precision * 100:.2f}')
    print(f'Recall: {recall * 100:.2f}')
    print(f'F1-score: {f1 * 100:.2f}')

    best_model = grid.best_estimator_
    print(f'BEST MODEL: {best_model}')
    print(f'BEST SCORE: {grid.best_score_ * 100:.2f}')
    print(f'BEST PARAMS: {grid.best_params_}')

def Grid_Search_Naive_Bayes(x_train_, y_train_, x_test_, y_test_):
    model = BernoulliNB()

    param_grid = {
        "alpha": [0.01, 0.1, 0.5, 1.0],   # hệ số smoothing Laplace
        "binarize": [None, 0.0, 0.5, 1.0],          # ngưỡng để chuyển thành dữ liệu nhị phân
        "fit_prior": [True, False]                  # có học prior từ dữ liệu hay không
    }
    grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'f1_weighted', cv = 4, verbose = 2)
    grid.fit(x_train_, y_train_)
    y_predict_ = grid.predict(x_test_)
    precision = precision_score(y_test_, y_predict_, average = 'weighted')
    recall = recall_score(y_test_, y_predict_, average = 'weighted')
    f1 = f1_score(y_test_, y_predict_, average = 'weighted')

    print(f'Precision: {precision * 100:.2f}')
    print(f'Recall: {recall * 100:.2f}')
    print(f'F1-score: {f1 * 100:.2f}')

    best_model = grid.best_estimator_
    print(f'BEST MODEL: {best_model}')
    print(f'BEST SCORE: {grid.best_score_ * 100:.2f}')
    print(f'BEST PARAMS: {grid.best_params_}')