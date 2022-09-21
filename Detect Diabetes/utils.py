import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
df=pd.read_csv("data/diabetes.csv")

# code to create distributions
def distributions(df):
    df.hist()
    plt.tight_layout()
    plt.show()
    
# create single distributions functions
def histograms(df):
    for col in df.select_dtypes(exclude='O').columns:
        import seaborn as sns
        print('Distribution of column', col)
        sns.distplot(df[col])
        plt.show()
        print('*********************************************')
        
        
# metrics evaluation
def evaluate(df,y_pred, y_test, model_name):
    print("Metrics for model", model_name)
    print("Accuracy score is ", metrics.accuracy_score(y_pred, y_test))
    print('')
    print("f1 score is ", metrics.f1_score(y_pred, y_test))
    print('')
    print("ROC-AUC score is ", metrics.roc_auc_score(y_pred, y_test))
    print('')
    print("Classification Report is ", metrics.classification_report(y_pred, y_test))
    print('')
