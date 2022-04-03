import pandas as pd
import numpy as np
import pickle

# helper functions


def process_csv_file():

    dtype = {
    "Unnamed: 0": "int64",
    "batch": "int64",
    "customer": "string",
    "payment_type": "string",
    "gender": "string",
    "name": "string",
    "phone": "string",
    "cust_acc": "string",
    "merchant": "string",
    "category": "string",
    "amount": "float64",
    "fraud": "int64",
    }
    parse_dates = ["chck_date"]
    # Read csv file

    data = pd.read_csv(
    "../data/mollie_fraud_dataset.csv", sep=";", dtype=dtype, parse_dates=parse_dates) 
    data.rename({"Unnamed: 0": "transaction_id"}, axis=1, inplace=True)

    strcol_list = ['customer', 'payment_type', 'gender', 'merchant', 'category']
    for col in strcol_list:
        data[col] = data[col].apply(lambda x: x[:-1])

    # fill null value for account type with previously known account type of customer
    data["cust_acc"] = data.groupby(["customer"])["cust_acc"].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Empty")
    )

    # create country variable and drop unnecessary columns
    data['country'] = data['cust_acc'].apply(lambda x: x[0:2])
    data.drop(['name', 'phone', 'cust_acc'], axis=1, inplace=True)

    data_df = data.loc[data['chck_date'].notna()]
    # save as pickled file for later use
    data_df.to_pickle("../data/df_ml.pkl")

    return data_df

   