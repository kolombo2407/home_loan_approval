import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def preclean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    delete Loan_ID column as unsufficient
    replace Y and N in target variable Loan_Status with 1 and 0
    :param df:
    :return: df without Loan_ID column
    '''
    df.drop('Loan_ID', axis=1, inplace=True)
    df['Loan_Status'].replace({'Y': 1, 'N': 0}, inplace=True)
    return df


def cat_num_splitter(df: pd.DataFrame) -> list:
    """
    return two lists with categorical and numerical columns' names
    :param df:
    :return: two lists
    """
    num_attr = list(df.select_dtypes(exclude='object'))
    cat_attr = list(df.select_dtypes(include='object'))

    # let's move from num_attr columns where number of unique values less than 12
    # and treat them as categorical
    transfer_val = []
    for n in num_attr:
        if (df[n].unique().shape[0]) < 12:
            cat_attr.append(n)
            transfer_val.append(n)

    for i in transfer_val:
        num_attr.remove(i)

    return num_attr, cat_attr


def outliers_cleaner(df: pd.DataFrame, col: list) -> pd.DataFrame:
    """
    removing outliers in dataframe
    :param df: dataframe
    :param col: list of numerical columns in dataframe
    :return: df with less variance in data
    """
    for i in col:
        q1 = np.percentile(df[i], 25)
        q3 = np.percentile(df[i], 75)
        iqr = q3 - q1
        # find idx for elements which bigger than Q3 + 1.5 * IQR
        out_idx = df[df[i] > q3 + 1.5 * iqr].index
        # as soon as our outliers places above the top limit, let's delete one sided outliers
        df.drop(out_idx, axis=0, inplace=True)
    return df