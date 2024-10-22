import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

FILE_PATH = 'data.xlsx'
CONTINUOUS_COLUMNS = ['Quantity', 'Price', 'Value', 'DoneVolume', 'DoneValue']
DISCRETE_COLUMNS = ['AccID', 'AccCode', 'SecID', 'SecCode', 'Exchange', 'Destination', 'OrderGiver', 'OrderTakerUserCode', 'OriginOfOrder']

def load_csv(file_path):
    return pd.read_csv(file_path)

def load_excel(file_path):
    return pd.read_excel(file_path)

def detect_continuous_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    df['Z_Score'] = (df[column] - mean) / std
    outliers = df[np.abs(df['Z_Score']) > threshold]
    return outliers

def continuous_outlier_analysis(df, columns, threshold=3):
    outlier_dict = {}
    new_columns = df.columns.tolist().append('Cause')
    continuous_outlier_df = pd.DataFrame(columns=new_columns)
    for column in columns:
        print(f"Analyzing column: {column}")
        outliers = detect_continuous_outliers_zscore(df, column, threshold)
        outlier_dict[column] = outliers
        temp_df = pd.DataFrame(columns=df.columns.tolist(), data=outliers)
        temp_df['Cause'] = column
        continuous_outlier_df = pd.concat([continuous_outlier_df, temp_df])
    continuous_outlier_df.drop_duplicates()
    return outlier_dict, continuous_outlier_df

def discrete_outliers_frequency(df, column, threshold_ratio):
    value_counts = df[column].value_counts(normalize=True)
    outliers = value_counts[value_counts < threshold_ratio]
    outlier_rows = df[df[column].isin(outliers.index)]
    return outliers, outlier_rows

def discrete_outlier_analysis(df, columns, threshold_ratio=0.001):
    outlier_dict = {}
    new_columns = df.columns.tolist()
    new_columns.append('Cause')
    discrete_outlier_df = pd.DataFrame(columns=new_columns)
    for column in columns:
        print(f"Analyzing column: {column}")
        outliers, outlier_rows = discrete_outliers_frequency(df, column, threshold_ratio)
        outlier_dict[column] = {'outliers': outliers, 'outlier_rows': outlier_rows}
        temp_df = pd.DataFrame(columns=df.columns.tolist(), data=outlier_rows)
        temp_df['Cause'] = column
        discrete_outlier_df = pd.concat([discrete_outlier_df, temp_df])
    discrete_outlier_df.drop_duplicates()
    return outlier_dict, discrete_outlier_df

def show_scatterplots(df):
    sns.scatterplot(x='Price', y='Quantity', hue='SecCode', 
                    data=df[df['anomaly'] == 1], palette='Set2', legend='full')
    sns.scatterplot(x='Price', y='Quantity', data=df[df['anomaly'] == -1],
        color='red', marker='X', s=100, label='Anomalies')
    plt.title('Anomaly Detection in Financial Data: Price vs Quantity')
    plt.xlabel('Price')
    plt.ylabel('Quantity')
    plt.legend(title='SecCode', loc='upper right')
    plt.show()

def isolation_forest(df):
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    subset = df[all_columns]
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(features)
    features['anomaly'] = model.predict(features)
    subset['anomaly'] = features['anomaly']
    anomalies = features[features['anomaly'] == -1]
    print(f"Number of Anomalies Detected: {len(anomalies)}")
    print(anomalies)
    show_scatterplots(subset)
    return anomalies

def main():
    df = load_excel(FILE_PATH)

    print('\n\n------------------------')
    print("Continuous Outlier Analysis Results")
    print('------------------------\n')

    continuous_outliers, continuous_outlier_df = continuous_outlier_analysis(df.copy(), CONTINUOUS_COLUMNS)

    for column, outlier_data in continuous_outliers.items():
        if not outlier_data.empty:
            print(f"Outliers in {column}:")
            print(outlier_data)
        else:
            print(f"No outliers in {column}")

    print('\n\n------------------------')
    print("Discrete Outlier Analysis Results")
    print('------------------------\n')
    discrete_outliers, discrete_outlier_df = discrete_outlier_analysis(df.copy(), DISCRETE_COLUMNS)

    for column, outlier_data in discrete_outliers.items():
        if not outlier_data['outliers'].empty:
            print(f"Outliers in {column}:")
            print(f"Outlier values and their frequencies:\n{outlier_data['outliers']}")
            print(f"Rows with outliers in {column}:\n{outlier_data['outlier_rows']}")
        else:
            print(f"No outliers in {column}")

    print('\n\n------------------------')
    print("Isolation Forest Results")
    print('------------------------\n')
    isolation_forest_results = isolation_forest(df.copy())
    print(isolation_forest_results.index)

main()