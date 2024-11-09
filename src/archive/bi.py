# ============================================================
# Section: Letting users choose the chart type (Deprecated)
# ============================================================

# chart_type_1 = container1.selectbox(label="Chart type", options=["Line Chart", "Pie Chart"], index=0, label_visibility="collapsed")
# chart_type_2 = container2.selectbox(label="Chart type", options=["Line Chart", "Pie Chart"], index=1, label_visibility="collapsed")

# match chart_type_1:
#     case "Line Chart": create_line_chart(container1, "1")
#     case "Pie Chart": create_pie_chart(container1, "1")

# match chart_type_2:
#     case "Line Chart": create_line_chart(container2, "2")
#     case "Pie Chart": create_pie_chart(container2, "2")


# col1, col2, col3 = st.columns(3)  # Columns are containers
# # overview_container = st.container(border=True, key="overview_container")
# metrics_container = st.container(key="metrics_container")


# overview_container = col1.container(border=True)
# overview_expander = overview_container.expander("More Details")
# overview_expander.write("ZM insert your report here")


# with col1:


#     # with overview_container:
#     with metrics_container:
#         col_a, col_b, col_c = st.columns(3)
#         with col_a:
#             st.metric("Total Number of Orders", "5K")

#         with col_b:
#             st.metric("Total Volume of Orders", "40.39M")

#         with col_c:
#             st.metric("Total Value of Orders", "$33.28M")
#     with st.expander("More Details"):
#         st.write('''
#             ZM insert your report here.
#         ''')

# st.header("Selections")
# st.write(f"Your selections are {st.session_state.selections}.")

# def columns_array_creator(total_number, max_column_per_row):
#     columns_array = []
#     count = 0
#     while(count + max_column_per_row < total_number):
#         count += max_column_per_row
#         columns_array.append(max_column_per_row)
#     if(total_number - count > 0):
#         columns_array.append(total_number - count)
#     return columns_array

# selections = st.session_state.selections
# unique_values = set(item.split(" - ")[0] for item in selections)
# unique_count = len(unique_values)

# outer_columns_array = columns_array_creator(unique_count, 2)
# st.write(outer_columns_array)


# def create_sankey(
#     parent_container,
#     key,
#     src_options=["SecCode", "Exchange"],
#     src_index=0,
#     target_options=["BuySell", "Destination"],
#     target_index=0,
# ):
#     # Create child container
#     child_container = parent_container.container()

#     # Create selection boxes
#     source = child_container.selectbox(
#         label="Source",
#         options=src_options,
#         index=src_index,
#         key=key + "sankey_source",
#         label_visibility="collapsed",
#     )

#     target = child_container.selectbox(
#         label="Target",
#         options=target_options,
#         index=target_index,
#         key=key + "sankey_target",
#         label_visibility="collapsed",
#     )

#     # Filter data based on source selection
#     if source == "SecCode":
#         top5_sec_code = df["SecCode"].value_counts().nlargest(5).index
#         df_filtered = df[df["SecCode"].isin(top5_sec_code)]
#     else:
#         df_filtered = df

#     # Create aggregation based on selected source and target
#     agg_df = df_filtered.groupby([source, target]).size().reset_index(name="Count")

#     # Prepare data for Sankey diagram
#     labels = list(agg_df[source].unique()) + list(df_filtered[target].unique())

#     # Create source index mapping
#     source_index = {code: idx for idx, code in enumerate(agg_df[source].unique())}

#     # Initialize lists for Sankey data
#     sources = []
#     targets = []
#     values = []

#     # Fill source, target, and values
#     for _, row in agg_df.iterrows():
#         sources.append(source_index[row[source]])
#         targets.append(labels.index(row[target]))
#         values.append(row["Count"])

#     # Create Sankey diagram
#     fig = go.Figure(
#         go.Sankey(
#             node=dict(
#                 pad=15,
#                 thickness=20,
#                 line=dict(color="black", width=0.5),
#                 label=labels
#             ),
#             link=dict(
#                 source=sources,
#                 target=targets,
#                 value=values
#             )
#         )
#     )

#     # Display the chart
#     child_container.plotly_chart(fig, key=key + "sankey")

#     # Optional automated report section
#     if 'enable_automated_report' in globals() and enable_automated_report:
#         expander = child_container.expander("View automated report")
#         expander.write(generate_automated_report(fig))