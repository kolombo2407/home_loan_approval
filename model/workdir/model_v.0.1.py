import os
import pandas as pd
import numpy as np
import scipy
import pickle
import psycopg2

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

connection_string = 'postgresql://kolombo2407:Banbyf-wanteh-mugda2@146.190.116.255:5432/hla_db'


def download_train_data(conn_string: str) -> pd.DataFrame:
    # downloading data from db
    db = create_engine(connection_string)
    conn = db.connect()
    return pd.read_sql('SELECT * FROM train_data', conn)


data = download_train_data(connection_string)


def preclean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    delete Loan_ID column as unsufficient
    replace Y and N in target variable Loan_Status with 1 and 0
    :param df:
    :return: df without Loan_ID column
    '''
    df.drop('Loan_ID', axis=1, inplace=True)
    df['Loan_Status'].replace({'Y': 1, 'N': 0})
    return df


def cat_num_splitter(df: pd.DataFrame) -> list:
    """
    return two lists with categorical and numerical columns' names
    :param df:
    :return: two lists
    """
    num_attr = list(df.select_dtypes(exclude='object'))
    cat_attr = list(df.select_dtypes(include='object'))
    for i in ['Loan_Amount_Term', 'Credit_History']:
        cat_attr.append(i)
        num_attr.remove(i)
    return num_attr, cat_attr
