"""
#read csv file
import pandas as pd
def read_csv_file(file_path):

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
#display dataframe
def display_dataframe(df):
    if df is not None:
        print(df.head())
    else:
        print("DataFrame is None")  

df = read_csv_file("../creditcard.csv")
display_dataframe(df)"""

