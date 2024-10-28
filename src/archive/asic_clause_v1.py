# Display selected chart
if selected_chart == "RG 265.130 (Best Execution Obligation)":
    st.subheader(selected_chart)
    # Assuming "Time In Force" is the "TimeInForce" column in the provided schema
    area_data = df.groupby(['TimeInForce', 'AccCode'])['CumulativeNumberOfOrders'].sum().reset_index()
    area_fig = px.area(area_data, x="TimeInForce", y="CumulativeNumberOfOrders", color="AccCode",
                    title=selected_chart,
                    labels={"TimeInForce": "Time In Force", "CumulativeNumberOfOrders": "Cumulative Number of Orders"})
    st.plotly_chart(area_fig)
    
elif selected_chart == "RG 265.12 (Supervision of Market Participants)":
    st.subheader(selected_chart)
    # Using DoneVolume and Value columns for summarization by AccCode
    summary_data = df.groupby('AccCode').agg({
        'CumulativeDoneVolume': 'last',  # Assuming cumulative sum is already in `CumulativeDoneVolume`
        'CumulativeValue': 'last'  # Assuming cumulative sum is in `CumulativeValue`
    }).reset_index()
    summary_data.columns = ['AccCode', 'Total Done Volume', 'Total Cumulative Value']
    st.dataframe(summary_data)
    
elif selected_chart == "RG 265.51 (Suspicious Activity Reporting)":
    st.subheader(selected_chart)
    # User input for large order threshold
    threshold = st.number_input("Threshold for large orders (number of orders):", min_value=1, value=10)

    # Classify orders based on the threshold
    df_with_classification = classify_orders_based_on_threshold(df.copy(), threshold)

    # Group by the new classification and AccCode
    bar_data = df_with_classification.groupby(['OrderSizeCategory', 'AccCode'])['DoneVolume'].sum().reset_index()
    bar_fig = px.bar(bar_data, x="AccCode", y="DoneVolume", color="OrderSizeCategory", 
                    barmode='group', title="RG 265.51 (Suspicious Activity Reporting) by Order Size",
                    labels={"DoneVolume": "Total Done Volume", "AccCode": "Account Code", "OrderSizeCategory": "Order Size"})
    st.plotly_chart(bar_fig)
                
elif selected_chart == "RG 265.74 (Crossing Systems Reporting)":
    st.subheader(selected_chart)
    # Treemap using OrderCapacity and AccCode, counting the number of orders
    tree_data = df.groupby(['ExecutionVenueCategory', 'AccCode']).size().reset_index(name='Count of Orders')
    tree_fig = px.treemap(tree_data, path=['ExecutionVenueCategory', 'AccCode'], values='Count of Orders', 
                        title=selected_chart,
                        labels={"ExecutionVenueCategory": "Execution Venue", "AccCode": "Account Code", "Count of Orders": "Number of Orders"})
    st.plotly_chart(tree_fig)