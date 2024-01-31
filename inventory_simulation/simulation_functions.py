import numpy as np
import simpy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default='notebook'
import pandas as pd
import random

INITIAL_STOCK = 250

DAILY_DEMAND_MEAN = 55
DAILY_DEMAND_STD = 5
INTER_ARRIVAL_TIME = 1

ORDER_QUANTITY = 250
SAFETY_STOCK = 500

QUANTITY_DEVIATION_MEAN = 0
QUANTITY_DEVIATION_STD = 15

LEAD_TIME_MEAN = 3
LEAD_TIME_STD = 0
ALLOWED_DELIVERY_DAY = 4

SIM_LENGTH = 365

COST_ORDER_PER_ITEM = 0
COST_PER_ORDER = 0
COST_HOLDING_PER_ITEM = 0.5 #in €

QUALITY_BOOKINGS = DAILY_DEMAND_MEAN * 0.8
QUALITY_PROBABILITY = 1 #in Percent


class InventorySystem:
    def __init__(self, env, policy):
        # initialize values
        self.policy = policy
        self.level = INITIAL_STOCK
        self.safety_stock = SAFETY_STOCK
        self.order_size = ORDER_QUANTITY
        self.last_change = 0.
        self.ordering_cost = 0.
        self.shortage_cost = 0.
        self.holding_cost = 0.
        self.history = [(0., INITIAL_STOCK, 0)]
        self.orderDetails = [(0, 0, 0, "initial", "initial")]
        # launch processes by first reviewing current inventory and then substracting demand for production
        env.process(self.review_inventory(env))
        env.process(self.use_material_in_production(env))

    def create_stochastic_good_entrance(self, env, units):
        """ Draw randomly from normal distribution for goods entrance quantity """
        random.seed(env.now)
        distribution_order_quantity = random.choice(np.random.normal(units + QUANTITY_DEVIATION_MEAN, QUANTITY_DEVIATION_STD, 1000))
        #never go below zero
        distribution_order_quantity = max(distribution_order_quantity, 0)
        return distribution_order_quantity

    def create_stochastic_arrival(self, env):
        """ Draw randomly for stochastic lead time """
        random.seed(env.now)
        leadTimeList = [LEAD_TIME_MEAN -1, LEAD_TIME_MEAN, LEAD_TIME_MEAN + 1]
        distribution_lead_time = random.choices(leadTimeList, weights=(10, 80, 10), k=1)[0]
        return distribution_lead_time     


    def place_order(self, env, units):
        """ Place and receive orders """
        # update ordering costs
        self.ordering_cost += (COST_PER_ORDER
                              + units * COST_ORDER_PER_ITEM)
        # determine when order will be received
        lead_time = LEAD_TIME_MEAN
        # Generate Shipping ID
        shipping_id = f"id-{env.now}"
        # Track upcoming order
        self.orderDetails.append((env.now, env.now + lead_time, units, shipping_id, "Order"))
        # Wait until lead_time
        stochastic_lead_time = self.create_stochastic_arrival(env)
        yield env.timeout(stochastic_lead_time)
        # update inventory level and costs
        self.update_cost(env)
        # Get the quantity delivery
        goods_entrance_quantity = self.create_stochastic_good_entrance(env, units)
        self.orderDetails.append((env.now, env.now, goods_entrance_quantity, shipping_id, "Wareneingang", ))
        # Add goods entrance to stock level
        self.level += goods_entrance_quantity
        self.last_change = env.now
        self.history.append((env.now, self.level, goods_entrance_quantity))

    def check_delivery_allowance(self, env):
        """ Check if order can be made at current environment point """
        if (env.now + LEAD_TIME_MEAN) % ALLOWED_DELIVERY_DAY == 0:
            return True
        else:
            return False

    def orderPolicySafetyStock(self, env):
        """ Order if current stock level is below safety stock """
        if self.level <= self.safety_stock:
            return True, self.order_size
        else:
            return False, self.order_size
    
    def orderPolicyFutureDemand(self, env):
        """ Order if current stock level + quantity in transition cannot meet demand until next delivery """        
        #Time & Demand until next delivery
        demand_duration = LEAD_TIME_MEAN + ALLOWED_DELIVERY_DAY
        demand_cover = demand_duration * DAILY_DEMAND_MEAN

        #Get orders ins transit
        df_orders = pd.DataFrame(self.orderDetails)
        #This line prevents that we expect an order which has already arrived because of the stochastic lead time
        #We drop the rows which have duplicates (Order and Wareneingang)
        df_orders = df_orders.drop_duplicates(subset=[3], keep=False)
        df_relevant_orders = df_orders[
                (df_orders[1] > env.now) &
                (df_orders[1] < env.now + demand_cover)
            ].reset_index(drop=True)       
        orders_in_transit = df_relevant_orders[2].sum()

        #print(f"Zeitpunkt: {env.now}, Inventory: {self.level}, Transit: {orders_in_transit}, Demand: {demand_cover}")      

        if (self.level + orders_in_transit - demand_cover) <= self.safety_stock:
            return True, self.safety_stock + demand_cover - self.level - orders_in_transit
        else:
            return False, self.order_size     

    def review_inventory(self, env):
        """ Check inventory level at regular intervals and place
        orders inventory level is below reorder point """
        while True:
            #Check for Delivery Allowance
            orderingAllowed = self.check_delivery_allowance(env)
            if orderingAllowed:
                if self.policy == "safety_stock":
                    orderingNecessary, orderingQuantity = self.orderPolicySafetyStock(env)
                elif self.policy == "future_demand":
                    orderingNecessary, orderingQuantity = self.orderPolicyFutureDemand(env)
                else:
                    raise ValueError("Policy not included")
            else:
                orderingNecessary = False
                orderingQuantity = self.order_size

            # place order if required
            if (orderingNecessary & orderingAllowed):
                env.process(self.place_order(env, orderingQuantity))
            else:
                self.history.append((env.now, self.level, 0))   
            # wait for next check
            yield env.timeout(1.0)

    def update_cost(self, env):
        """ Update holding and shortage cost at each inventory
         movement """
        # update shortage cost
        if self.level <= 0:
            shortage_cost = 1000000
            self.shortage_cost += shortage_cost
        else:
            # update holding cost
            holding_cost = (self.level
                            * COST_HOLDING_PER_ITEM
                            * (env.now - self.last_change))
            self.holding_cost += holding_cost

    def create_demands(self, env, demand, std):
        """ Draw randomly from normal distribution """
        random.seed(env.now)
        distribution_demand = random.choice(np.random.normal(demand, std, 1000))
        return distribution_demand

    def create_quality_bookings(self, env, quality_booking, quality_booking_prob):
        """ Draw randomly from Quality Bookings """
        random.seed(env.now)
        numberList = [quality_booking, 0]
        quality_booking = random.choices(numberList, weights=(quality_booking_prob, 100-quality_booking_prob), k=1)[0]
        return quality_booking     

    def use_material_in_production(self, env):
        """ Substract Demand from current inventory level """
        while True:
            # generate next demand size and time
            iat = INTER_ARRIVAL_TIME
            size = self.create_demands(env, DAILY_DEMAND_MEAN, DAILY_DEMAND_STD)
            # generate quality issues
            quality_issues = self.create_quality_bookings(env, QUALITY_BOOKINGS, QUALITY_PROBABILITY)
            yield env.timeout(iat)
            # update inventory level and costs upon demand receipt
            self.update_cost(env)
            self.level -= size + quality_issues
            self.last_change = env.now
            self.history.append((env.now, self.level, 0))


def step_graph(inventory):
    """ Displays a step line chart of inventory level """
    # create subplot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.grid(which = 'major', alpha = .4)
    # plot simulation data
    x_val = [x[0] for x in inventory.history]
    y_val = [x[1] for x in inventory.history]
    plt.step(x_val, y_val, where = 'post', label='Units in inventory')
    # titles and legends
    plt.xlabel('Months')
    plt.ylabel('Units in inventory')
    plt.gca().legend()
    plt.show()

def step_graph_plotly(inventory):
    """ Displays a step line chart of inventory level with plotly """
    x_values = [x[0] for x in inventory.history]
    y_values = [x[1] for x in inventory.history]
    o_values = [x[2] for x in inventory.history]
    df = pd.DataFrame(list(zip(x_values, y_values, o_values)), columns =['Days', 'Inventory', 'Orders'])
    fig = px.line(df, x="Days", y="Inventory", title='Simulation Results')
    fig.show(renderer="notebook")
    return df

def run(display_chart:bool = True, policy:str = "safety_stock"):
    """
    Run Inventory simulation
    """
    # check user inputs
    if SIM_LENGTH <= 0:
        raise ValueError("Simulation length must be greater than zero")
    
    # setup simulation
    env = simpy.Environment()
    inv = InventorySystem(env, policy)
    env.run(SIM_LENGTH)
    # compute and return simulation results
    avg_total_cost = (inv.ordering_cost 
                    + inv.holding_cost 
                    + inv.shortage_cost) / SIM_LENGTH
    if display_chart == True:
        dataframe = step_graph(inv)    
    return avg_total_cost, inv, dataframe