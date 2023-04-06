import os
import pandas as pd
import numpy as np
import scipy
import pickle
import psycopg2

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine

from data_preprocessing import cat_num_splitter, preclean_data, outliers_cleaner

connection_string = 'postgresql://kolombo2407:Banbyf-wanteh-mugda2@146.190.116.255:5432/hla_db'


def download_train_data(conn_string: str) -> pd.DataFrame:
    # downloading data from db
    db = create_engine(connection_string)
    conn = db.connect()
    return pd.read_sql('SELECT * FROM train_data', conn)


data = download_train_data(connection_string)

data = preclean_data(data)

num_attr, cat_attr = cat_num_splitter(data.drop('Loan_Status', axis=1))

data = outliers_cleaner(data, num_attr)

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

num_pipe = Pipeline([
    ('num_imp', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('std_scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('cat_imp', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first'))
])

transform_pipe = ColumnTransformer([
    ('num', num_pipe, num_attr),
    ('cat', cat_pipe, cat_attr)
])

X_prepared = transform_pipe.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_prepared,
                                                    y,
                                                    random_state=24,
                                                    test_size=.2)

dec_tree_clf = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(estimator=dec_tree_clf, random_state=24)

param_grid = {
    'n_estimators': [10, 20, 30, 100, 150, 200, 300]
}

grid_search = GridSearchCV(estimator=bagging_clf, param_grid=param_grid)

grid_search.fit(X_train, y_train)

y_scored = grid_search.predict(X_test)

precision, recall, threshold = precision_recall_curve(y_test, y_scored)

pr_auc = auc(recall, precision)
f1 = f1_score(y_test, y_scored)

print(pr_auc, f1)
