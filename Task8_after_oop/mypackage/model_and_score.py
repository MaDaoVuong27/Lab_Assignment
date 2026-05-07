import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

def Model(self, x_train_final, y_train_final, x_test_final, y_test_final, type):

    score_dict = {}

    #train
    for id, model_ in self.models.items():
        print(model_)
        model_.fit(x_train_final, y_train_final)
        y_predict_final = model_.predict(x_test_final)

        precision = round(precision_score(y_test_final, y_predict_final, average = 'weighted') * 100, 2) 
        recall = round(recall_score(y_test_final, y_predict_final, average = 'weighted') * 100, 2)
        f1 = round(f1_score(y_test_final, y_predict_final, average = 'weighted') * 100, 2)

        score_dict[id] = [precision, recall, f1]

        """
        if(type == 0):
            print(classification_report(y_test_final, y_predict_final, digits = 4))
        else:
            print(classification_report(y_test_final, y_predict_final, digits = 4, target_names = self.multi_classes_))
        """
    df_score_dict = pd.DataFrame.from_dict(score_dict, orient = 'index', columns = ['Precision', 'Recall', 'F1']) #orient = index: keys của dict sẽ trở thành index (hàng) của DataFrame.
    print(df_score_dict)




