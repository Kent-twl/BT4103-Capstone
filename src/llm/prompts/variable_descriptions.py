## Descriptions of variables in the dataset
variable_prompt = """
Here are brief explanations of the variables that you may encounter.
{variables}
"""

##TODO: Split the prompt into variables for different contexts (BI, ASIC, Anomaly Detection) and exclude irrelevant variables
variable_descriptions = """
Instance: Terminal used to submit the order.
OrderNo: Unique identifier for the order.
ParentOrderNo: If an order is only partially fulfilled, the remaining part is stored as a new order with a new OrderNo and the original order is stored as a parent order.
RootParentOrderNo: Used to identify the original order in a chain of orders. Can be equal to OrderNo.
CreateDate: Date and time when the order was created.
DeleteDate: Date and time when the order was deleted.
AccID: Account ID of the user who submitted the order.
AccCode: Account code of the user who submitted the order.
BuySell: B = Buy, S = Sell.
Side: Buy or sell side. 1 = Buy, 2 = Sell.
OrderSide: Buy, Long Sell or Short Sell.
SecID: Unique identifier for the security.
SecCode: Security code.
Exchange: Exchange where the order was submitted.
Destination: Where the order was sent to, can be equal to Exchange.
Quantity: Number of securities in the order.
PriceMultiplier: Multiplier for the price.
Price: Price of the security.
Value: Total value of the order.
ValueMultiplier: Multiplier for the value.
DoneVolume: Volume of the order that has been executed.
DoneValue: Value of the order that has been executed.
Currency: Currency of the order.
OrderType: Similar to BuySell.
PriceInstruction: Limit or market order.
TimeInForce: Period of time for which the order is active.
Lifetime: Intended order duration, e.g. good till cancelled (GTC), end of day (EOD).
ClientOrderID: Client order identifier.
SecondaryClientOrderID: Secondary client order identifier.
DestOrderNo: Destination order number.
ExtDestOrderNo: External destination order number.
DestUserID: Destination user ID.
OrderGiver: The giver of the order, used in compliance checks.
OrderTakerUserCode: The user code of the order taker.
IntermediaryID: For agency orders, identifies an intermediary that provided instructions for the order.
OriginOfOrder: For agency orders, identifies the person who provided instructions for the order. 
OrderCapacity: Capacity in which a market participant has submitted an order (principal, agent, etc.).
DirectedWholesale: Indicates whether the order was submitted by a wholesale client.
ExecutionVenue: The venue (licensed market, crossing system, etc.) where the order was executed.
ExecutionInstructionsRaw: Raw execution instructions.
"""

## Dict mapping variable to its description
variable_dict = {}
for line in variable_descriptions.strip().split('\n'):
    if line:
        variable, description = line.split(': ', 1)
        variable_dict[variable] = f"{variable}: {description}"