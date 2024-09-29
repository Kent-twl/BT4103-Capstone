import pandas as pd
import numpy as np

FILE_PATH = 'data.csv'
CONTINUOUS_COLUMNS = ['Quantity', 'Price', 'Value', 'DoneVolume', 'DoneValue']
DISCRETE_COLUMNS = ['AccID', 'AccCode', 'SecID', 'SecCode', 'Exchange', 'Destination', 'OrderGiver', 'OrderTakerUserCode', 'OriginOfOrder']

def load_csv(file_path):
    return pd.read_csv(file_path)

def detect_continuous_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['Z_Score'] = (df[column] - mean) / std
    outliers = df[np.abs(df['Z_Score']) > threshold]
    return outliers

def continuous_outlier_analysis(df, columns, threshold=3):
    outlier_dict = {}
    for column in columns:
        print(f"Analyzing column: {column}")
        outliers = detect_continuous_outliers_zscore(df, column, threshold)
        outlier_dict[column] = outliers
    return outlier_dict

def discrete_outliers_frequency(df, column, threshold_ratio=0.05):
    value_counts = df[column].value_counts(normalize=True)
    outliers = value_counts[value_counts < threshold_ratio]
    outlier_rows = df[df[column].isin(outliers.index)]
    return outliers, outlier_rows

def discrete_outlier_analysis(df, columns, threshold_ratio=0.05):
    outlier_dict = {}
    for column in columns:
        print(f"Analyzing column: {column}")
        outliers, outlier_rows = discrete_outliers_frequency(df, column, threshold_ratio)
        outlier_dict[column] = {'outliers': outliers, 'outlier_rows': outlier_rows}
    return outlier_dict

def main():
    df = load_csv(FILE_PATH)

    continuous_outliers = continuous_outlier_analysis(df, CONTINUOUS_COLUMNS)

    for column, outlier_data in continuous_outliers.items():
        print(f"Outliers in {column}:")
        print(outlier_data)


    discrete_outliers = discrete_outlier_analysis(df, DISCRETE_COLUMNS)

    for column, data in discrete_outliers.items():
        print(f"Outliers in {column}:")
        print(f"Outlier values and their frequencies:\n{data['outliers']}")
        print(f"Rows with outliers in {column}:\n{data['outlier_rows']}")

main()