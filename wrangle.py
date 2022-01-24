import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# import splitting
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def prepare_prop(prop_df):
    
    
    # We will delete these columns from DataFrames as they have low influence in sale price of the property
    # Unnamed: 0, BUILDING CLASS CATEGORY, BLOCK, LOT, EASE-MENT, ADDRESS, APARTMENT NUMBER, BUILDING CLASS AT PRESENT 

    prop_df = prop_df.drop(columns = ['Unnamed: 0', 'BUILDING CLASS CATEGORY', 'BLOCK', 'LOT', 'EASE-MENT', 'ADDRESS', 'APARTMENT NUMBER', 'BUILDING CLASS AT PRESENT'])
    
    # Replacing white space in column head with '_' as it will be easy to deal with column head later on.
    prop_df.columns= prop_df.columns.str.replace(' ', '_')

    # Converting column head to lower string
    prop_df.columns= prop_df.columns.str.lower()

    # getting rid of unneccessary while space in column head
    prop_df.columns = prop_df.columns.str.strip()
    
    # Replacing '-' value with empty in sale price
    prop_df.sale_price = prop_df.sale_price.str.replace("-", "")

    # Replacing '-' value with empty in land square feet
    prop_df.land_square_feet = prop_df.land_square_feet.str.replace("-", "")

    # Replacing '-' value with empty in gross square feet
    prop_df.gross_square_feet = prop_df.gross_square_feet.str.replace("-", "")
    
    # Replacing one or more space characeters  with Null
    prop_df = prop_df.replace(r'^\s*$', np.nan, regex=True)
    
    # Dropping NAN from DataFrame
    prop_df = prop_df.dropna()
    
    
    # Conveting sale price to float data type
    prop_df.sale_price = prop_df.sale_price.astype(float)
    
    # Conveting land square feet to float data type
    prop_df.land_square_feet = prop_df.land_square_feet.astype(float)
    
    # Conveting gross square feet to float data type
    prop_df.gross_square_feet = prop_df.gross_square_feet.astype(float)
    
    # Removing record for which sale price is equal to or less than or equal to $1000
    prop_df = prop_df[prop_df.sale_price > 1000]
    
    # Removing record for which sale price is equal to or less than or equal to $1000
    prop_df = prop_df[prop_df.gross_square_feet > 0]

    # Removing record for which sale price is equal to or less than or equal to $1000
    prop_df = prop_df[prop_df.land_square_feet > 0]
    
    # age column created by subtracting 2022 - year_built
    prop_df['age'] = 2022 - prop_df.year_built
    
    # dropping year_built
    prop_df = prop_df.drop(columns = ['year_built'])
    
    col_list = ['total_units', 'land_square_feet', 'gross_square_feet', 'sale_price', 'age']
    
    # Removing outliers
    prop_df = remove_outliers(prop_df, 1.5, col_list)
    
    # return train validate test
    # train_validate, test = train_test_split(prop_df, test_size=.2, random_state=123)
    # train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return prop_df

def split_prop_df(prop_df):
    """ Splits the prod_df into train, validate and test"""
    
    train_validate, test = train_test_split(prop_df, test_size = 0.2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = 0.3, random_state = 123)
    
    return train, validate, test
