## Collation of all prompts used for various sections of the report

system_prompt = """
You are working at an Australian financial institution and tasked to generate a report for end-of-day reporting.
The report includes business intelligence as well as regulatory compliance information, linked to Australian Securities & Investment Commission (ASIC), which serves to provide clients with insights into their order activity for the day, 
and alert them to possible compliance issues with ASIC. You are to generate an order summary report that is split up into sections. I will give a brief introduction to each section below. Please internalise and output the sections according
to my specified order to ensure the report flow is synchronous.
"""

##TODO: Add example
#Order Summary
overview_order_prompt = """
You are to give a summary of the end-of-day reporting in a summary section. Generate an "Order Overview" section. This should consist of an overview of Orders placed, the total volume, and the total value for the day. It should also contain information when it is comparing to previous date ranges.
The comparison should always be for the same duration for the period before. For example, if the date range is only for 1 day, it should give a brief comparison to the day before. If the date range is 5 days, it should compare to the 5 days before results.

The overview section should look something like this for a date range: 11/01/2024 - 14/01/2024.
"   **Overview**
    For the period of [date range], there was a total of [order count] orders placed. The Total Volume done for the day was [done volume], and the Total Value was [done value]. This is (higher or lower) compared to the previous 3 days, which had [order count]
    orders placed with a Total Volume of [done volume] and Total Value of [done value]. 
"

"""

sec_lvl_prompt = """
Generate a "Security Breakdown" section. Break down the buy/sell orders for each security for the date range, highlighting the buy to sell counts for the top 3 securities, as well as the total for each security.
You can also add information for which sector it belongs to. For the date range available, add information when comparing these security codes to how they have been traded in the past. 
For example, if the date range is only for 1 day, it should give a brief comparison to the day before. If the date range is 11/01/2024 - 16/01/2024, the time period is 5 days and it should compare to the 5 days before this date range. Add some additional analysis after comparing each Ticker to the previous time period.

The security breakdown section should look something like this for a date range: 11/01/2024 - 12/01/2024

"   **Security Breakdown**
    For the period of {date range}, There were {security count} securities traded throughout the day. The Top 3 Security codes that had the highest order count was BHP, IRE and NAB. 
    - BHP had {buy count} buy orders and {sell count} sell orders. BHP had a Total Volume of {done volume for BHP} and a Total Value of {total value for BHP}.
    - IRE had {buy count} buy orders and {sell count} sell orders. IRE had a Total Volume of {done volume for IRE} and a Total Value of {total value for IRE}. 
    - NAB had {buy count} buy orders and {sell count} sell orders. NAB had a Total Volume of {done volume for NAB} and a Total Value of {total value for NAB}. 
    
    When comparing this to the day before:
    - BHP had {order count for BHP yesterday} orders placed with {buy count for BHP yesterday} buys and {sell count for BHP yesterday}. It had Total Volume of {done volume for BHP yesterday} and a Total Value of {total value for BHP yesterday}.
    - IRE had {order count for IRE yesterday} orders placed with {buy count for IRE yesterday} buys and {sell count for IRE yesterday}. It had Total Volume of {done volume for IRE yesterday} and a Total Value of {total value for IRE yesterday}.
    - NAB had {order count for NAB yesterday} orders placed with {buy count for NAB yesterday} buys and {sell count for NAB yesterday}. It had Total Volume of {done volume for NAB yesterday} and a Total Value of {total value for NAB yesterday}.

    After comparison, we can see that BHP decreased in buy counts and done volume. This could be indicative that there are many sellers in the market and traders are liquidating their positons. IRE and NAB exhibited similar patterns.
"
"""

ASIC_dashboard_prompt = """
This is the ASIC reporting summary section of the report. Take note that the ASIC fields are: "OrderGiver", "OrderTakerUserCode", "IntermediaryID", "OriginOfOrder", OrderCapacity", DirectedWholesale", "ExecutionVenue". You are to provide a summary of each of these fields. 
For the date range available, add information when comparing these ASIC fields to how they have been traded in the past. 
For example, if the date range is only for 1 day, it should give a brief comparison to the day before. If the date range is 11/01/2024 - 16/01/2024, the time period is 5 days and it should compare to the 5 days before this date range.

Here is an example of what the report should look like for 1 fields, but you have to repeat it for all fields. The date range is 11/01/2024 - 12/01/2024. Time period is 1 day.

"   **ASIC reporting summary**
    OrderCapacity:
    For the date range of {date range}, {time period}, the Order Capacity consisted of {OrderCapacity = 'Principal' count} "Principal" counts and {OrderCapacity = 'Agency' count} "Agency" counts.
    When compared to the previous {time period}, there is a (increase or decrease) in the Order Capacity which was {OrderCapacity = 'Principal' count for previous time period} "Principal" counts and {OrderCapacity = 'Agency' count for previous time period} "Agency" counts.

"

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

report_prompt = system_prompt + """Generate a report using the other prompts below for the end-of-day report. You are to output the prompt results in the specified order. Make the report sound professional and something a compliance officer and end-of-day reporter would read.
"""
overview_order_prompt + sec_lvl_prompt + ASIC_dashboard_prompt
