import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Name of data file
FILE_PATH = 'data.xlsx'
# Relevant columns for continuous and discrete outlier analysis
CONTINUOUS_COLUMNS = ['Quantity', 'Price', 'Value', 'DoneVolume', 'DoneValue']
DISCRETE_COLUMNS = ['AccCode', 'BuySell', 'Side', 'OrderSide', 'SecCode', 'Exchange', 'Destination', 'Lifetime', 'OrderGiver', 'OrderTakerUserCode']
# Proportion of anomalies to identify
PROPORTION = 0.01

# Function to load data
def load_excel(file_path):
    return pd.read_excel(file_path)

# Function to identify continuous outliers given a specific column
def detect_continuous_outliers_zscore(df, column, threshold=3):
    # Get mean and sd
    mean = df[column].mean()
    std = df[column].std()
    # Calculate z score
    df['zScoreOfCause'] = (df[column] - mean) / std
    # Identify outliers (z score more than 3)
    outliers = df[np.abs(df['zScoreOfCause']) > threshold]
    return outliers

# Function to conduct continuous outlier analysis on relevant columns
def continuous_outlier_analysis(df, columns, threshold=3):
    # Dictionary to store all outliers by column
    outlier_dict = {}
    new_columns = df.columns.tolist().append('Cause')
    # Store all outliers in a dataframe
    continuous_outlier_df = pd.DataFrame(columns=new_columns)
    for column in columns:
        print(f"Analyzing column: {column}")
        # Identify outliers for each column
        outliers = detect_continuous_outliers_zscore(df, column, threshold)
        outlier_dict[column] = outliers
        temp_df = pd.DataFrame(columns=df.columns.tolist(), data=outliers)
        # Keep track of cause of outlier
        temp_df['Cause'] = column
        continuous_outlier_df = pd.concat([continuous_outlier_df, temp_df])
    continuous_outlier_df.drop_duplicates()
    return outlier_dict, continuous_outlier_df

# Function to identify discrete outliers given a specific column
def discrete_outliers_frequency(df, column, threshold_ratio):
    # Check distribution of data
    value_counts = df[column].value_counts(normalize=True)
    # Identify outliers (proportion of discrete value lower than a certain threshold)
    outliers = value_counts[value_counts < threshold_ratio]
    outlier_rows = df[df[column].isin(outliers.index)]
    return outliers, outlier_rows

# Function to conduct discrete outlier analysis on relevant columns
def discrete_outlier_analysis(df, columns, threshold_ratio=0.001):
    # Dictionary to store all outliers by column
    outlier_dict = {}
    new_columns = df.columns.tolist()
    new_columns.append('Cause')
    # Store all outliers in a dataframe
    discrete_outlier_df = pd.DataFrame(columns=new_columns)
    for column in columns:
        print(f"Analyzing column: {column}")
        # Identify outliers for each column
        outliers, outlier_rows = discrete_outliers_frequency(df, column, threshold_ratio)
        outlier_dict[column] = {'outliers': outliers, 'outlier_rows': outlier_rows}
        temp_df = pd.DataFrame(columns=df.columns.tolist(), data=outlier_rows)
        # Keep track of cause of outlier
        temp_df['Cause'] = column
        discrete_outlier_df = pd.concat([discrete_outlier_df, temp_df])
    discrete_outlier_df.drop_duplicates()
    return outlier_dict, discrete_outlier_df

# Function to display scatterplot of price vs quantity, with anomalies marked out
def show_scatterplots(df, title, position, axs):
    # Data points coloured by security code
    sns.scatterplot(x='Price', y='Quantity', hue='SecCode', 
                    data=df[df['anomaly'] == 1], palette='Set2', ax=axs[position])
    # Represent anomalies using a red X
    sns.scatterplot(x='Price', y='Quantity', data=df[df['anomaly'] == -1],
        color='red', marker='X', s=100, label='Anomalies', ax=axs[position])
    axs[position].legend().set_visible(False)
    plt.title(f'Anomaly Detection in Financial Data: Price vs Quantity')
    plt.xlabel('Price')
    plt.ylabel('Quantity')
    # plt.show(block=False)

# Function to run the isolation forest model
def isolation_forest(df, axs):
    print("Running isolation forest...")
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    # Filter the data to all relevant columns
    subset = df[all_columns]
    # One hot encoding for discrete columns
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    # Create model
    model = IsolationForest(n_estimators=100, contamination=PROPORTION * 3, random_state=42)
    # Fit model to data
    model.fit(features)
    # Predict which points are anomalies
    features['anomaly'] = model.predict(features)
    subset['anomaly'] = features['anomaly']
    # Identify anomalies
    anomalies = features[features['anomaly'] == -1]
    # Determine features causing the anomalies to be flagged out
    anomalies = get_top_features_if(model, features, anomalies)
    print(f"Number of IF Anomalies Detected: {len(anomalies)}")
    # show_scatterplots(subset, "Isolation Forest", 0, axs)
    return anomalies

# Function to run the one-class SVM model
def ocsvm(df, axs):
    print("Running one-class SVM...")
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    # Filter the data to all relevant columns
    subset = df[all_columns]
    # One hot encoding for discrete columns
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    # Normalize the data (beneficial for one-class SVM)
    scaler = StandardScaler()
    features[CONTINUOUS_COLUMNS] = scaler.fit_transform(features[CONTINUOUS_COLUMNS])
    # Create model
    model = svm.OneClassSVM(nu=PROPORTION, kernel="rbf", gamma=0.01) 
    # Fit model to data
    model.fit(features)
    # Predict which points are anomalies
    features['anomaly'] = model.predict(features)
    subset['anomaly'] = features['anomaly']
    # Identify anomalies
    anomalies = features[features['anomaly'] == -1]
    print(f"Number of OCSVM Anomalies Detected: {len(anomalies)}")
    # show_scatterplots(subset, "One-Class SVM", 1, axs)
    return anomalies

# Function to run the gaussian mixture model
def gmm(df, axs):
    print("Running gaussian mixture model...")
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    # Filter the data to all relevant columns
    subset = df[all_columns]
    # One hot encoding for discrete columns
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    # Create model
    model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    # Fit model to data
    model.fit(features)
    # Predict which points are anomalies
    log_likelihood = model.score_samples(features)
    features['log_likelihood'] = log_likelihood
    threshold = np.percentile(log_likelihood, PROPORTION * 100)
    # Identify anomalies
    anomalies = features[features['log_likelihood'] < threshold]
    features['anomaly'] = np.where(features['log_likelihood'] < threshold, -1, 1)
    subset['anomaly'] = features['anomaly']
    print(f"Number of GMM Anomalies Detected: {len(anomalies)}")
    # show_scatterplots(subset, "Gaussian Mixture Model", 2, axs)
    return anomalies

# Function to identify the top 3 features that are causing a datapoint to be anomalous
def get_top_features_if(model, features, anomalies):
    # Conduct SHAP analysis
    explainer = shap.Explainer(model, features)
    shap_values = explainer(features)
    top_feature_1 = []
    top_feature_2 = []
    top_feature_3 = []
    # Get top 3 features for each anomaly
    for i in anomalies.index:
        shap_value_anomaly = shap_values[i]       
        # Sort by contributing factor
        top_3_features = np.argsort(np.abs(shap_value_anomaly.values))[-3:]   
        # Keep track of each     
        top_feature_1.append(features.columns[top_3_features[-1]])
        top_feature_2.append(features.columns[top_3_features[-2]])
        top_feature_3.append(features.columns[top_3_features[-3]])
    # Append all to the dataframe as new columns
    anomalies['top_feature_1'] = top_feature_1
    anomalies['top_feature_2'] = top_feature_2
    anomalies['top_feature_3'] = top_feature_3
    return anomalies

# Function to get the anomalies identified across different algorithms
def get_common_anomalies(df, results1, results2, results3):
    # Datapoint is classified as anomalous if it is flagged by isolation forest and minimally one other algorithm
    common_indexes = results1.index.intersection(results2.index.union(results3.index))
    df = df.loc[common_indexes] 
    # Append the corresponding top features as new columns
    df = df.merge(results1[['top_feature_1', 'top_feature_2', 'top_feature_3']], left_index=True, right_index=True, how='left')
    return df

# Function to conduct outlier analysis
def outlier_results(df):
    print("Running outlier analysis...")
    # Continuous outlier analysis
    continuous_outliers, continuous_outlier_df = continuous_outlier_analysis(df.copy(), CONTINUOUS_COLUMNS)
    # Discrete outlier analysis
    discrete_outliers, discrete_outlier_df = discrete_outlier_analysis(df.copy(), DISCRETE_COLUMNS)
    return continuous_outlier_df, discrete_outlier_df

# Function to conduct anomaly detection
def anomaly_results(df):
    print("Running anomaly detection...")
    # Isolation forest anomaly detection
    if_results = isolation_forest(df.copy(), axs=None)
    # One-class SVM anomaly detection
    ocsvm_results = ocsvm(df.copy(), axs=None)
    # Gaussian mixture model anomaly detection
    gmm_results = gmm(df.copy(), axs=None)
    # Combine results
    overall_anomaly_df = get_common_anomalies(df.copy(), if_results, ocsvm_results, gmm_results)
    return overall_anomaly_df

# Main function to be used for standalone anomaly detection, not to be used with dashboard
def main():
    # Load data
    df = load_excel(FILE_PATH)

    # Continuous outlier analysis
    print('\n\n------------------------')
    print("Continuous Outlier Analysis Results")
    print('------------------------\n')

    continuous_outliers, continuous_outlier_df = continuous_outlier_analysis(df.copy(), CONTINUOUS_COLUMNS)

    # Print results of continuous outlier analysis for each column
    for column, outlier_data in continuous_outliers.items():
        if not outlier_data.empty:
            print(f"Outliers in {column}:")
            print(outlier_data)
        else:
            print(f"No outliers in {column}")

    # Discrete outlier analysis
    print('\n\n------------------------')
    print("Discrete Outlier Analysis Results")
    print('------------------------\n')

    discrete_outliers, discrete_outlier_df = discrete_outlier_analysis(df.copy(), DISCRETE_COLUMNS)

    # Print results of discrete outlier analysis for each column
    for column, outlier_data in discrete_outliers.items():
        if not outlier_data['outliers'].empty:
            print(f"Outliers in {column}:")
            print(f"Outlier values and their frequencies:\n{outlier_data['outliers']}")
            print(f"Rows with outliers in {column}:\n{outlier_data['outlier_rows']}")
        else:
            print(f"No outliers in {column}")

    # Plot for displaying results of the 3 anomaly detection algorithms
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Isolation forest anomaly detection
    print('\n\n------------------------')
    print("Isolation Forest Results")
    print('------------------------\n')
    if_results = isolation_forest(df.copy(), axs)
    print(if_results.index)

    # One-class SVM anomaly detection
    print('\n\n------------------------')
    print("One-Class SVM Results")
    print('------------------------\n')
    ocsvm_results = ocsvm(df.copy(), axs)
    print(ocsvm_results.index)

    # Gaussian mixture model anomaly detection
    print('\n\n------------------------')
    print("Gaussian Mixture Model Results")
    print('------------------------\n')
    gmm_results = gmm(df.copy(), axs)
    print(gmm_results.index)

    # Display results of individual algorithms
    plt.tight_layout()
    plt.show()


    print('\n\n------------------------')
    print("Anomaly Detection Results")
    print('------------------------\n')
    # Combined results of anomaly detection
    overall_anomaly_results = get_common_anomalies(df.copy(), if_results, ocsvm_results, gmm_results)
    print(overall_anomaly_results)

    return continuous_outlier_df, discrete_outlier_df, overall_anomaly_results