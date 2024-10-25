## Script for calculating metrics to be included in reports (similar to those on dashboard)
import pandas as pd
from datetime import datetime
from pprint import pprint


def calculate_metrics(df_data, config=None):
    ## Pre-processing
    df_data['BuySell'] = df_data['BuySell'].replace(['B'], 'Buy')
    df_data['BuySell'] = df_data['BuySell'].replace(['S'], 'Sell')

    ##TODO: Update using config file
    calculation_date = datetime(2024, 2, 1)  # Can be loaded in via a configuration file
    historical_start_date = datetime(2024, 2, 1)  # Can be loaded in via a configuration file
    historical_end_date = datetime(2024, 2, 1)  # Can be loaded in via a configuration file
    historical_num_of_days = (historical_end_date - historical_start_date).days + 1

    calculation_data = df_data[df_data['CreateDate'].dt.floor('D') == calculation_date]
    historical_data = df_data[
        (df_data['CreateDate'].dt.floor('D') >= historical_start_date) & 
        (df_data['CreateDate'].dt.floor('D') <= historical_end_date)
    ]

    ## Metrics
    bi_viz_1_overview = __bi_viz_1_overview(calculation_data, historical_data, historical_num_of_days)
    # pprint(bi_viz_1_overview)

    bi_viz_2_sector = __bi_viz_2_sector(df_data, calculation_data, historical_data, historical_num_of_days)
    # pprint(bi_viz_2_sector)

    bi_viz_3_capacity = __bi_viz_3_capacity(calculation_data, historical_data, historical_num_of_days)
    # pprint(bi_viz_3_capacity)

    bi_viz_4_lifetime = __bi_viz_4_lifetime(df_data, calculation_data, historical_data, historical_num_of_days)
    # pprint(bi_viz_4_lifetime)

    bi_viz_5_price_instruction = __bi_viz_5_price_instruction(df_data, calculation_data, historical_data, historical_num_of_days)
    # pprint(bi_viz_5_price_instruction)

    bi_viz_6_trades_over_time = None ## Placeholder
    # bi_viz_6_trades_over_time = __bi_viz_6_trades_over_time(df_data)
    # pprint(bi_viz_6_trades_over_time)

    bi_viz_7_sankey_diagram = __bi_viz_7_sankey_diagram(df_data)
    # pprint(bi_viz_7_sankey_diagram)

    
    ##TODO: Return required metrics based on config file
    return bi_viz_1_overview, bi_viz_2_sector, bi_viz_3_capacity, bi_viz_4_lifetime, \
        bi_viz_5_price_instruction, bi_viz_6_trades_over_time, bi_viz_7_sankey_diagram


## Helper function to convert string to datetime object
def str_to_datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

## Helper function to calculate the percentage difference between two values
def calculate_percentage_difference(current, historical):
    if historical == 0:  # Avoid division by zero
        return float('inf')  # or handle it as needed
    return round(((current - historical) / historical) * 100, 2)


## Metrics that correspond to BI visualization 1: Overview
def __bi_viz_1_overview(calculation_data, historical_data, historical_num_of_days):
    bi_viz_1_overview = {
        'current': {
            'total_number_of_orders': calculation_data.shape[0],
            'total_volume_of_orders': calculation_data['DoneVolume'].sum(),
            'total_value_of_orders': round((calculation_data['Value'] * calculation_data['ValueMultiplier']).sum(), 2),
        },
        'average_historical': {
            'total_number_of_orders': int(historical_data.shape[0] / historical_num_of_days),
            'total_volume_of_orders': int(historical_data['DoneVolume'].sum() / historical_num_of_days),
            'total_value_of_orders': round(((historical_data['Value'] * historical_data['ValueMultiplier']).sum()) / historical_num_of_days, 2),
        },
        'percentage_difference': {
            'in_total_orders': calculate_percentage_difference(
                calculation_data.shape[0],
                int(historical_data.shape[0] / historical_num_of_days)
            ),
            'in_total_volume': calculate_percentage_difference(
                calculation_data['DoneVolume'].sum(),
                int(historical_data['DoneVolume'].sum() / historical_num_of_days)
            ),
            'in_total_value': calculate_percentage_difference(
                round((calculation_data['Value'] * calculation_data['ValueMultiplier']).sum(), 2),
                round(((historical_data['Value'] * historical_data['ValueMultiplier']).sum()) / historical_num_of_days, 2)
            ),
        }
    }
    return bi_viz_1_overview


## Metrics that correspond to BI visualization 2: Sector
def __bi_viz_2_sector(df_data, calculation_data, historical_data, historical_num_of_days):
    bi_viz_2_sector = {}

    for sec in df_data['SecCode'].unique():
        bi_viz_2_sector[sec] = {
            'current': {
                'number_of_buy_orders': calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts().get('Buy', 0),
                'percentage_of_buy_orders': round(calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Buy', 0) * 100, 2),
                'number_of_sell_orders': calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts().get('Sell', 0),
                'percentage_of_sell_orders': round(calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Sell', 0) * 100, 2),
                'total_number_of_orders': calculation_data[calculation_data['SecCode'] == sec].shape[0],
            },
            'average_historical': {
                'number_of_buy_orders': int(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts().get('Buy', 0) / historical_num_of_days),
                'percentage_of_buy_orders': round(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Buy', 0) * 100, 2),
                'number_of_sell_orders': historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts().get('Sell', 0),
                'percentage_of_sell_orders': round(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Sell', 0) * 100, 2),
                'total_number_of_orders': int(historical_data[historical_data['SecCode'] == sec].shape[0] / historical_num_of_days),
            },
            'percentage_difference': {
                'in_number_of_buy_orders': calculate_percentage_difference(
                    calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts().get('Buy', 0),
                    int(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts().get('Buy', 0) / historical_num_of_days)
                ),
                'in_percentage_of_buy_orders': calculate_percentage_difference(
                    round(calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Buy', 0) * 100, 2),
                    round(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Buy', 0) * 100, 2)
                ),
                'in_number_of_sell_orders': calculate_percentage_difference(
                    calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts().get('Sell', 0),
                    int(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts().get('Sell', 0))
                ),
                'in_percentage_of_sell_orders': calculate_percentage_difference(
                    round(calculation_data[calculation_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Sell', 0) * 100, 2),
                    round(historical_data[historical_data['SecCode'] == sec]['BuySell'].value_counts(normalize=True).get('Sell', 0) * 100, 2)
                ),
                'in_total_number_of_orders': calculate_percentage_difference(
                    calculation_data[calculation_data['SecCode'] == sec].shape[0],
                    int(historical_data[historical_data['SecCode'] == sec].shape[0] / historical_num_of_days)
                ),
            }
        }

    return bi_viz_2_sector


## Metrics that correspond to BI visualization 3: Capacity
def __bi_viz_3_capacity(calculation_data, historical_data, historical_num_of_days):
    bi_viz_3_capacity = {'current': {}, 'average_historical': {}, 'percentage_difference': {}}
    capacity_types = ['Agency', 'Principal']
    counts = calculation_data['OrderCapacity'].value_counts()
    historical_counts = historical_data['OrderCapacity'].value_counts()
    
    for capacity in capacity_types:
        # Current data calculations
        bi_viz_3_capacity['current'][f'number_of_orders_with_{capacity.lower()}_capacity'] = counts.get(capacity, 0)
        bi_viz_3_capacity['current'][f'percentage_of_orders_with_{capacity.lower()}_capacity'] = round(counts.get(capacity, 0) / len(calculation_data) * 100, 2)

        # Historical data calculations
        bi_viz_3_capacity['average_historical'][f'number_of_orders_with_{capacity.lower()}_capacity'] = int(historical_counts.get(capacity, 0) / historical_num_of_days)
        bi_viz_3_capacity['average_historical'][f'percentage_of_orders_with_{capacity.lower()}_capacity'] = round(historical_counts.get(capacity, 0) / len(historical_data) * 100, 2)

        # Percentage difference calculations
        bi_viz_3_capacity['percentage_difference'][f'in_number_of_orders_with_{capacity.lower()}_capacity'] = calculate_percentage_difference(
            bi_viz_3_capacity['current'][f'number_of_orders_with_{capacity.lower()}_capacity'],
            bi_viz_3_capacity['average_historical'][f'number_of_orders_with_{capacity.lower()}_capacity']
        )
        bi_viz_3_capacity['percentage_difference'][f'in_percentage_of_orders_with_{capacity.lower()}_capacity'] = calculate_percentage_difference(
            bi_viz_3_capacity['current'][f'percentage_of_orders_with_{capacity.lower()}_capacity'],
            bi_viz_3_capacity['average_historical'][f'percentage_of_orders_with_{capacity.lower()}_capacity']
        )
    
    return bi_viz_3_capacity


## Metrics that correspond to BI visualization 4: Lifetime
def __bi_viz_4_lifetime(df_data, calculation_data, historical_data, historical_num_of_days):
    bi_viz_4_lifetime = {'current': {}, 'average_historical': {}, 'percentage_difference': {}}
    lifetime_types = df_data['Lifetime'].unique()
    counts = calculation_data['Lifetime'].value_counts()
    historical_counts = historical_data['Lifetime'].value_counts()
    
    for lifetime in lifetime_types:
        # Current data calculations
        bi_viz_4_lifetime['current'][f'number_of_orders_with_{lifetime.lower()}_lifetime'] = counts.get(lifetime, 0)
        bi_viz_4_lifetime['current'][f'percentage_of_orders_with_{lifetime.lower()}_lifetime'] = round(counts.get(lifetime, 0) / len(calculation_data) * 100, 2)

        # Historical data calculations
        bi_viz_4_lifetime['average_historical'][f'number_of_orders_with_{lifetime.lower()}_lifetime'] = int(
            historical_counts.get(lifetime, 0) / historical_num_of_days
        )
        bi_viz_4_lifetime['average_historical'][f'percentage_of_orders_with_{lifetime.lower()}_lifetime'] = round(
            historical_counts.get(lifetime, 0) / len(historical_data) * 100, 2
        )

        # Percentage difference calculations
        bi_viz_4_lifetime['percentage_difference'][f'in_number_of_orders_with_{lifetime.lower()}_lifetime'] = calculate_percentage_difference(
            bi_viz_4_lifetime['current'][f'number_of_orders_with_{lifetime.lower()}_lifetime'],
            bi_viz_4_lifetime['average_historical'][f'number_of_orders_with_{lifetime.lower()}_lifetime']
        )
        bi_viz_4_lifetime['percentage_difference'][f'in_percentage_of_orders_with_{lifetime.lower()}_lifetime'] = calculate_percentage_difference(
            bi_viz_4_lifetime['current'][f'percentage_of_orders_with_{lifetime.lower()}_lifetime'],
            bi_viz_4_lifetime['average_historical'][f'percentage_of_orders_with_{lifetime.lower()}_lifetime']
        )
    
    return bi_viz_4_lifetime


## Metrics that correspond to BI visualization 5: Price instruction
def __bi_viz_5_price_instruction(df_data, calculation_data, historical_data, historical_num_of_days):
    bi_viz_5_price_instruction = {'current': {}, 'average_historical': {}, 'percentage_difference': {}}
    price_instruction_types = df_data['PriceInstruction'].unique()
    counts = calculation_data['PriceInstruction'].value_counts()
    historical_counts = historical_data['PriceInstruction'].value_counts()

    for pi in price_instruction_types:
        # Current data calculations
        bi_viz_5_price_instruction['current'][f'number_of_orders_with_{pi.lower()}_price_instruction'] = counts.get(pi, 0)
        bi_viz_5_price_instruction['current'][f'percentage_of_orders_with_{pi.lower()}_price_instruction'] = round(
            counts.get(pi, 0) / len(calculation_data) * 100, 2
        )

        # Historical data calculations
        bi_viz_5_price_instruction['average_historical'][f'number_of_orders_with_{pi.lower()}_price_instruction'] = int(
            historical_counts.get(pi, 0) / historical_num_of_days
        )
        bi_viz_5_price_instruction['average_historical'][f'percentage_of_orders_with_{pi.lower()}_price_instruction'] = round(
            historical_counts.get(pi, 0) / len(historical_data) * 100, 2
        )

        # Percentage difference calculations
        bi_viz_5_price_instruction['percentage_difference'][f'in_number_of_orders_with_{pi.lower()}_price_instruction'] = calculate_percentage_difference(
            bi_viz_5_price_instruction['current'][f'number_of_orders_with_{pi.lower()}_price_instruction'],
            bi_viz_5_price_instruction['average_historical'][f'number_of_orders_with_{pi.lower()}_price_instruction']
        )
        bi_viz_5_price_instruction['percentage_difference'][f'in_percentage_of_orders_with_{pi.lower()}_price_instruction'] = calculate_percentage_difference(
            bi_viz_5_price_instruction['current'][f'percentage_of_orders_with_{pi.lower()}_price_instruction'],
            bi_viz_5_price_instruction['average_historical'][f'percentage_of_orders_with_{pi.lower()}_price_instruction']
        )
    
    return bi_viz_5_price_instruction


## Metrics that correspond to BI visualization 6: Trades over time
##TODO: Incomplete
def __bi_viz_6_trades_over_time(df_data):
    df_data['order_create_date_rounded_to_nearest_hour'] = df_data['CreateDate'].dt.floor('H')
    trade_counts = df_data.groupby('order_create_date_rounded_to_nearest_minute').size()
    trade_counts_df = trade_counts.reset_index(name='trade_count')
    trade_counts_df.sort_values(by='order_create_date_rounded_to_nearest_minute', inplace=True)
    total_trades = trade_counts_df['trade_count'].sum()
    trade_counts_df['percentage_of_total_trade_count'] = round((trade_counts_df['trade_count'] / total_trades) * 100, 2)
    trade_counts_df['absolute_change_of_trade_count_from_prev_minute'] = trade_counts_df['trade_count'].diff()
    
    return trade_counts_df


## Metrics that correspond to BI visualization 7: Sankey diagram
def __bi_viz_7_sankey_diagram(df_data):
    bi_viz_7_sankey_diagram = {}
    sector_types = df_data['SecCode'].unique()

    total_counts = df_data['SecCode'].value_counts(normalize=True)
    buy_counts = df_data[df_data['BuySell']=='Buy']['SecCode'].value_counts(normalize=True)
    sell_counts = df_data[df_data['BuySell']=='Sell']['SecCode'].value_counts(normalize=True)
    
    for sector in sector_types:
        bi_viz_7_sankey_diagram[sector] = {}
        bi_viz_7_sankey_diagram[sector][f'percentage_of_{sector}_accounts_in_total_orders'] = round(total_counts.get(sector, 0) * 100, 2)
        bi_viz_7_sankey_diagram[sector][f'percentage_of_{sector}_accounts_in_buy_orders'] = round(buy_counts.get(sector, 0) * 100, 2)
        bi_viz_7_sankey_diagram[sector][f'percentage_of_{sector}_accounts_in_sell_orders'] = round(sell_counts.get(sector, 0) * 100, 2)
    
    return bi_viz_7_sankey_diagram