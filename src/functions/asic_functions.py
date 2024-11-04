def classify_execution_venue(value):
    return 'Others' if None else 'Crossing Systems'

def classify_orders_based_on_threshold(df, threshold):
    df["OrderSizeCategory"] = df["CreateDate_CumulativeNumberOfOrders"].apply(lambda x: "Large Order" if x >= threshold else "Normal Order")
    return df