def load_csv(file_path):
    return pd.read_csv(file_path)

def get_top_features_svm(model, features, anomalies):
    def predict_fn_svm(X):
        return model.decision_function(X)
    explainer = shap.KernelExplainer(predict_fn_svm, features)
    shap_values = explainer.shap_values(features)
    anomalies = get_top_features(shap_values, features, anomalies)
    return anomalies

def get_top_features_gmm(model, features, anomalies):
    def predict_fn_gmm(X):
        return model.score_samples(X)
    explainer = shap.KernelExplainer(predict_fn_gmm, features)
    shap_values = explainer.shap_values(features)
    anomalies = get_top_features(shap_values, features, anomalies)
    return anomalies

def get_top_features(shap_values, features, anomalies):
    top_feature_1 = []
    top_feature_2 = []
    top_feature_3 = []
    for i in anomalies.index:
        shap_value_anomaly = shap_values[i]        
        top_3_features = np.argsort(np.abs(shap_value_anomaly))[-3:]
        top_feature_1.append(features.columns[top_3_features[-1]]) 
        top_feature_2.append(features.columns[top_3_features[-2]])
        top_feature_3.append(features.columns[top_3_features[-3]])
    anomalies['top_feature_1'] = top_feature_1
    anomalies['top_feature_2'] = top_feature_2
    anomalies['top_feature_3'] = top_feature_3
    return anomalies

def show_overall_scatterplot(df, anomalies, field):
    fig, ax = plt.subplots(figsize=(2,2))
    sns.scatterplot(x='Price', y='Quantity', hue=field,
                    data=df, palette='Set2', legend=False)
    sns.scatterplot(x='Price', y='Quantity', data=anomalies,
        color='red', marker='X', s=100, label='Anomalies')
    plt.title("Anomales Detected in Provided Data")
    ax.set_xlabel('Price')
    ax.set_ylabel('Quantity')
    return fig