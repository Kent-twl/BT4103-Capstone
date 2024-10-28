def classify_execution_venue(value):
    if 0 < value < 0.2:
        return 'Licensed Market'
    elif 0.2 <= value < 0.4:
        return 'Crossing System'
    elif 0.4 <= value < 0.6:
        return 'System Internaliser'
    elif 0.6 <= value < 0.8:
        return 'Over-the-Counter'
    else:
        return 'FX'

def classify_orders_based_on_threshold(df, threshold):
    df["OrderSizeCategory"] = df["CumulativeNumberOfOrders"].apply(lambda x: "Large Order" if x >= threshold else "Normal Order")
    return df