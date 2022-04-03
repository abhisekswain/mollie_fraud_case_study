import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from preprocess import process_csv_file



# function to classify datetime to weekday/weekend
def is_weekend(transaction_date):
    """computes if date is weekend
    Args:
       datetime variable
    Returns:
       boolean: is_weekend
    """
    # transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = transaction_date.weekday()
    # binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday >= 5

    return int(is_weekend)

# function to classify datetime to day/night
def is_night(transaction_date):
    """computes if time is at night
       defines as between 00:00 and 06:00
    Args:
       datetime variable
    Returns:
       boolean: is_night
    """
    # Get the hour of the transaction
    tx_hour = transaction_date.hour
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = tx_hour <= 6

    return int(is_night)

def get_customer_rfm_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
    """computes avg number of transactions
       and avg transactions amount for given window
    Args:
       df: dataframe with customer transactions
    Returns:
       df: dataframe with computed avg and num transactions
    """
    # let us first order transactions chronologically
    customer_transactions = customer_transactions.sort_values("chck_date")

    # transaction date and time is set as the index, which will allow the use of the rolling function
    customer_transactions.index = customer_transactions.chck_date

    # lloping through each each window size
    for window_size in windows_size_in_days:

        # compute the sum of the transaction amounts and the number of transactions for the given window size
        sum_amount_transaction_window = (
            customer_transactions["amount"].rolling(str(window_size) + "d").sum()
        )
        num_transaction_window = (
            customer_transactions["amount"].rolling(str(window_size) + "d").count()
        )

        # compute the average transaction amount for the given window size
        # num_transaction_window is always >0 since current transaction is always included
        avg_amount_transaction_window = (
            sum_amount_transaction_window / num_transaction_window
        )

        # save feature values in list and add to df
        customer_transactions[
            "customer_nb_tx" + str(window_size) + "day_window"
        ] = list(num_transaction_window)
        customer_transactions[
            "customer_avg_amount_" + str(window_size) + "day_window"
        ] = list(avg_amount_transaction_window)

    # teindex according to transaction IDs
    customer_transactions.index = customer_transactions.transaction_id

    # return the dataframe with the new features
    return customer_transactions
def get_count_risk_rolling_window(
    terminal_transactions,
    delay_period=7,
    windows_size_in_days=[1, 7, 30],
    feature="merchant",
):
    """computes a risk score for each merchant
       based on historical fraudulent transactions
    Args:
       df: dataframe with customer transactions
    Returns:
       computed risk score
    """
    terminal_transactions = terminal_transactions.sort_values("chck_date")

    terminal_transactions.index = terminal_transactions.chck_date

    num_fraud_delay = (
        terminal_transactions["fraud"].rolling(str(delay_period) + "d", closed= "left").sum()
    )
    num_transaction_delay = (
        terminal_transactions["fraud"].rolling(str(delay_period) + "d", closed= "left").count()
    )

    for window_size in windows_size_in_days:

        num_fraud_delay_window = (
            terminal_transactions["fraud"]
            .rolling(str(delay_period + window_size) + "d", closed= "left")
            .sum()
        )
        num_transaction_delay_window = (
            terminal_transactions["fraud"]
            .rolling(str(delay_period + window_size) + "d", closed= "left")
            .count()
        )

        num_fraud_window = num_fraud_delay_window - num_fraud_delay
        num_transaction_window = num_transaction_delay_window - num_transaction_delay

        risk_window = num_fraud_window / num_transaction_window

        terminal_transactions[
            feature + "_num_transactions_" + str(window_size) + "day_window"
        ] = list(num_transaction_window)
        terminal_transactions[
            feature + "_risk_" + str(window_size) + "day_window"
        ] = list(risk_window)

    terminal_transactions.index = terminal_transactions.transaction_id

    # replace NA values with 0 (all undefined risk scores where num_transaction_window is 0)
    terminal_transactions.fillna(0, inplace=True)

    return terminal_transactions

def transform_cat_feats(df):
    """makes null columns into unknown and cat columns
    are label encoded
    Args:
    df (pd.DataFrame): Dataframe with the transaction data.
    Returns:
    Dataframe with numerically encoded data for cat cols
    """

    cat = [
        "payment_type",
        "gender",
        "country",
        "category",
    ]

    # "batch"    ]

    for feature in cat:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])
        feat_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return df



def get_feats():
    # Let's read in the pickle dataframe
    #df_ml = pd.read_pickle("../data/df_ml.pkl")
    df_ml = process_csv_file()
    df_ml['is_weekend']=df_ml.chck_date.apply(is_weekend)
    df_ml['is_night']=df_ml.chck_date.apply(is_night)
    df_ml = df_ml.groupby("customer").apply(
    lambda x: get_customer_rfm_features(x, windows_size_in_days=[1, 7, 30]))
    df_ml = df_ml.sort_values("chck_date").reset_index(drop=True)
    df_ml = df_ml.groupby("merchant").apply(
    lambda x: get_count_risk_rolling_window(
        x, delay_period=7, windows_size_in_days=[1, 7, 30], feature="merchant"
    ))
    df_ml = df_ml.sort_values("chck_date").reset_index(drop=True)
    df_ml = transform_cat_feats(df_ml)
    # Save dataframe to use later
    df_ml.to_pickle("../data/df_final.pkl")
    print ("feature creation done")


if __name__ == "__main__":
    get_feats()
    #args = sys.argv[1:]
    #main(args)
