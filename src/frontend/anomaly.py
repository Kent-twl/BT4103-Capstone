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

FILE_PATH = 'data.xlsx'
CONTINUOUS_COLUMNS = ['Quantity', 'Price', 'Value', 'DoneVolume', 'DoneValue']
DISCRETE_COLUMNS = ['AccCode', 'BuySell', 'Side', 'OrderSide', 'SecCode', 'Exchange', 'Destination', 'Lifetime', 'OrderGiver', 'OrderTakerUserCode']
PROPORTION = 0.01

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

def show_scatterplots(df, title, position, axs):
    sns.scatterplot(x='Price', y='Quantity', hue='SecCode', 
                    data=df[df['anomaly'] == 1], palette='Set2', ax=axs[position])
    sns.scatterplot(x='Price', y='Quantity', data=df[df['anomaly'] == -1],
        color='red', marker='X', s=100, label='Anomalies', ax=axs[position])
    axs[position].legend().set_visible(False)
    # plt.title(f'Anomaly Detection in Financial Data: Price vs Quantity')
    plt.xlabel('Price')
    plt.ylabel('Quantity')
    # plt.legend(title='SecCode', loc='upper right')
    # plt.show(block=False)

def isolation_forest(df, axs):
    print("Running isolation forest...")
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    subset = df[all_columns]
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    model = IsolationForest(n_estimators=100, contamination=PROPORTION * 3, random_state=42)
    model.fit(features)
    features['anomaly'] = model.predict(features)
    subset['anomaly'] = features['anomaly']
    anomalies = features[features['anomaly'] == -1]
    anomalies = get_top_features_if(model, features, anomalies)
    print(f"Number of Anomalies Detected: {len(anomalies)}")
    # show_scatterplots(subset, "Isolation Forest", 0, axs)
    return anomalies

def ocsvm(df, axs):
    print("Running one-class SVM...")
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    subset = df[all_columns]
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    scaler = StandardScaler()
    features[CONTINUOUS_COLUMNS] = scaler.fit_transform(features[CONTINUOUS_COLUMNS])
    model = svm.OneClassSVM(nu=PROPORTION, kernel="rbf", gamma=0.01) 
    model.fit(features)
    features['anomaly'] = model.predict(features)
    subset['anomaly'] = features['anomaly']
    anomalies = features[features['anomaly'] == -1]
    print(f"Number of Anomalies Detected: {len(anomalies)}")
    # show_scatterplots(subset, "One-Class SVM", 1, axs)
    return anomalies

def gmm(df, axs):
    print("Running gaussian mixture model...")
    all_columns = DISCRETE_COLUMNS + CONTINUOUS_COLUMNS
    subset = df[all_columns]
    features = pd.get_dummies(subset, columns=DISCRETE_COLUMNS)
    model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    model.fit(features)
    log_likelihood = model.score_samples(features)
    features['log_likelihood'] = log_likelihood
    threshold = np.percentile(log_likelihood, PROPORTION * 100)
    anomalies = features[features['log_likelihood'] < threshold]
    features['anomaly'] = np.where(features['log_likelihood'] < threshold, -1, 1)
    subset['anomaly'] = features['anomaly']
    print(f"Number of Anomalies Detected: {len(anomalies)}")
    # show_scatterplots(subset, "Gaussian Mixture Model", 2, axs)
    return anomalies

def get_top_features_if(model, features, anomalies):
    explainer = shap.Explainer(model, features)
    shap_values = explainer(features)
    top_feature_1 = []
    top_feature_2 = []
    top_feature_3 = []
    for i in anomalies.index:
        shap_value_anomaly = shap_values[i]        
        top_3_features = np.argsort(np.abs(shap_value_anomaly.values))[-3:]        
        top_feature_1.append(features.columns[top_3_features[-1]])
        top_feature_2.append(features.columns[top_3_features[-2]])
        top_feature_3.append(features.columns[top_3_features[-3]])
    anomalies['top_feature_1'] = top_feature_1
    anomalies['top_feature_2'] = top_feature_2
    anomalies['top_feature_3'] = top_feature_3
    return anomalies

# def get_top_features_svm(model, features, anomalies):
#     def predict_fn_svm(X):
#         return model.decision_function(X)
#     explainer = shap.KernelExplainer(predict_fn_svm, features)
#     shap_values = explainer.shap_values(features)
#     anomalies = get_top_features(shap_values, features, anomalies)
#     return anomalies

# def get_top_features_gmm(model, features, anomalies):
#     def predict_fn_gmm(X):
#         return model.score_samples(X)
#     explainer = shap.KernelExplainer(predict_fn_gmm, features)
#     shap_values = explainer.shap_values(features)
#     anomalies = get_top_features(shap_values, features, anomalies)
#     return anomalies

# def get_top_features(shap_values, features, anomalies):
#     top_feature_1 = []
#     top_feature_2 = []
#     top_feature_3 = []
#     for i in anomalies.index:
#         shap_value_anomaly = shap_values[i]        
#         top_3_features = np.argsort(np.abs(shap_value_anomaly))[-3:]
#         top_feature_1.append(features.columns[top_3_features[-1]]) 
#         top_feature_2.append(features.columns[top_3_features[-2]])
#         top_feature_3.append(features.columns[top_3_features[-3]])
#     anomalies['top_feature_1'] = top_feature_1
#     anomalies['top_feature_2'] = top_feature_2
#     anomalies['top_feature_3'] = top_feature_3
#     return anomalies

def get_common_anomalies(df, results1, results2, results3):
    common_indexes = results1.index.intersection(results2.index.union(results3.index))
    df = df.loc[common_indexes] 
    df = df.merge(results1[['top_feature_1', 'top_feature_2', 'top_feature_3']], left_index=True, right_index=True, how='left')
    return df

def outlier_results(df):
    print("Running outlier analysis...")
    continuous_outliers, continuous_outlier_df = continuous_outlier_analysis(df.copy(), CONTINUOUS_COLUMNS)
    discrete_outliers, discrete_outlier_df = discrete_outlier_analysis(df.copy(), DISCRETE_COLUMNS)
    return continuous_outlier_df, discrete_outlier_df

def anomaly_results(df):
    print("Running anomaly detection...")
    if_results = isolation_forest(df.copy(), axs=None)
    ocsvm_results = ocsvm(df.copy(), axs=None)
    gmm_results = gmm(df.copy(), axs=None)
    overall_anomaly_df = get_common_anomalies(df.copy(), if_results, ocsvm_results, gmm_results)
    return overall_anomaly_df

def show_overall_scatterplot(df, anomalies, field):
    fig, ax = plt.subplots(figsize=(2,2))
    sns.scatterplot(x='Price', y='Quantity', hue=field,
                    data=df, palette='Set2', legend=False)
    sns.scatterplot(x='Price', y='Quantity', data=anomalies,
        color='red', marker='X', s=100, label='Anomalies')
    # ax.set_title('Anomaly Detection in Provided Data')
    plt.title("Anomales Detected in Provided Data")
    ax.set_xlabel('Price')
    ax.set_ylabel('Quantity')
    return fig

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

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    print('\n\n------------------------')
    print("Isolation Forest Results")
    print('------------------------\n')
    if_results = isolation_forest(df.copy(), axs)
    print(if_results.index)

    print('\n\n------------------------')
    print("One-Class SVM Results")
    print('------------------------\n')
    ocsvm_results = ocsvm(df.copy(), axs)
    print(ocsvm_results.index)

    print('\n\n------------------------')
    print("Gaussian Mixture Model Results")
    print('------------------------\n')
    gmm_results = gmm(df.copy(), axs)
    print(gmm_results.index)

    plt.tight_layout()
    plt.show()

    print('\n\n------------------------')
    print("Anomaly Detection Results")
    print('------------------------\n')
    overall_anomaly_results = get_common_anomalies(df.copy(), if_results, ocsvm_results, gmm_results)
    print(overall_anomaly_results)

    return continuous_outlier_df, discrete_outlier_df, overall_anomaly_results