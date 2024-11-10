# Archive Code to display old Dashboard Config
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

# Calculate the count and percentage of orders filled immediately
immediate_fill_count = filtered_df[filtered_df['ExecutionTime'] == 0]['CumulativeNumberOfOrders'].sum()
total_order_count = filtered_df['CumulativeNumberOfOrders'].sum()
immediate_fill_percentage = (immediate_fill_count / total_order_count) * 100 if total_order_count > 0 else 0

# Display the metrics with custom styling
st.markdown("""
    <div style="text-align: center; padding: 10px; border-radius: 8px; background-color: #4CAF50; color: white;">
        <h3>Orders Filled Immediately</h3>
        <h1 style="font-size: 40px;">{:.2f}%</h1>
        <p style="font-size: 18px;">({:,} out of {:,} orders)</p>
    </div>
""".format(immediate_fill_percentage, immediate_fill_count, total_order_count), unsafe_allow_html=True)

# Display a breakdown of orders by ExecutionTime within this specific AccCode
execution_time_data = filtered_df.groupby('ExecutionTime')['CumulativeNumberOfOrders'].sum().reset_index()
execution_time_fig = px.bar(
    execution_time_data, 
    x="ExecutionTime", 
    y="CumulativeNumberOfOrders",
    title="Number of Orders by Time Taken for Execution",
    labels={"ExecutionTime": "Execution Time", "CumulativeNumberOfOrders": "Number of Orders"}
)
st.plotly_chart(execution_time_fig)