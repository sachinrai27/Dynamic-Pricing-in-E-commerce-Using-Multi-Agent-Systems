class Product:
    def __init__(self, cogs, initial_price, base_demand):
        """
        Initializes the Product class.

        Args:
            cogs (float): Cost of goods sold (COGS) for the product.
            initial_price (float): Initial price of the product.
            base_demand (float): Base demand value for the product.
        """
        self.cogs = cogs
        self.current_price = initial_price
        self.base_demand = base_demand 