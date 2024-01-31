# Inventory Simulation
In this inventory simulation it is possible to model different risks which can affect the overall inventory levels. While it is quite easy to model a single risks it can get very complicated if you model multiple risks at once. The simulation basically consists of these main components:
* Modeled Risks: Delivery Delays, Delivery Quantity Deviations, Quality Deviations and Demand Risks
* An order process based on a certain policy - when and how much should be ordered
* Inbound process - Goods get deliverd
* Outbound process - Material is reduced in production due to demand
The simulation basically models how effective the provided order policy when facing these risks.

# Short intro into inventory management
"Inventory management is a discipline primarily about specifying the shape and placement of stocked goods. It is required at different locations within a facility or within many locations of a supply network to precede the regular and planned course of production and stock of materials." (https://en.wikipedia.org/wiki/Inventory)
This task actually sounds easier than it is -> how much inventory is good? Too much costs money, too less costs money aswell. This is and always be a trade-off and depends on the level of risk taking. In general it is advised to have as much stock as necessary but as low as possible (easy right?)

# How does simulation help?
Simulation != real life but it can help you to make changes in a "safer" way. Let's pretend you want to change the order policy of your product or you want to lower the safety stock. How could you know if it really works out? This is where a good simulation (not saying that this here is a good simulation) comes into play. You can check how your changes will most likely affect the whole inventory level.

# Where to go from here?
My actual goal was not to create a simulation. I took this simulation as a baseline for developing a `Reinforcement Learning Agent` who replaces basically the order policies. I am already working on it from time to time (and yes it already works in certain scenarios).
