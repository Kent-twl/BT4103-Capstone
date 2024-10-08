## Collation of all prompts used for various sections of the report

system_prompt = """
You are working at an Australian financial institution and tasked to generate a report for end-of-day reporting.
The report includes business intelligence as well as regulatory compliance information, linked to Australian Securities & Investment Commission (ASIC), which serves to provide clients with insights into their order activity for the day, 
and alert them to possible compliance issues with ASIC.
"""

##TODO: Add example

order_summary_prompt = """
Generate a trade summary for the end-of-day report. Take note that the ASIC fields are: "OrderGiver", "OrderTakerUserCode", "IntermediaryID", "OriginOfOrder", OrderCapacity", DirectedWholesale", "ExecutionVenue". The flow of the order summary report should be as followed:
1. This should start off with an overview of Orders placed, the total volume, and the total value for the day. 
2. Break down the buy/sell orders for each security, highlighting the buy to sell counts for the top 3 securities. You can also add information for which sector it belongs to.
3. Zoom into the ASIC reporting fields and provide a summary section of order statistics. For "OrderGiver", "OrderTakerUserCode", "IntermediaryID","OriginOfOrder" fields, provide the top 3 which placed the most number of orders. Also provide the statistics for those with the lowest.
4. In the next section will be business intelligence summary. Look at fields such as "Lifetime", "Price Instruction", "SecCode" and others, excluding the ASIC fields. Give a comprehensive overview and zoom into the details regarding trade statistics that will be useful to end-of-day reporting.
...
"""

anomaly_detection_prompt = system_prompt + """
You are tasked with writing the compliance section of the report. The orders have been run through an anomaly detection model.
Given the output, generate a report that highlights suspicious orders, and provide brief explanations for why they are suspicious.
You may reference the following rules that outline what constitutes a suspicious order:

a) ...
b) ...
c) ...
You may also reference some examples of suspicious orders and the explanations:
a) ...
b) ...
Please provide a report using the below output from the anomaly detection model:
... data...
"""