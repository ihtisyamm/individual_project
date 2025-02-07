import simpy
import random

class Bank:
    def __init__(self, env, num_tellers):
        self.env = env
        self.teller = simpy.Resource(env, num_tellers)
        
    def customer_arrival(self, customer_id):
        arrival_time = self.env.now
        print(f'Customer {customer_id} arrives at {arrival_time:.2f}')
        
        with self.teller.request() as request:
            # Wait for teller
            yield request
            
            # Start service
            wait_time = self.env.now - arrival_time
            print(f'Customer {customer_id} gets teller at {self.env.now:.2f} (waited {wait_time:.2f})')
            
            # Service time is random between 5-10 minutes
            service_time = random.uniform(5, 10)
            yield self.env.timeout(service_time)
            
            print(f'Customer {customer_id} leaves at {self.env.now:.2f}')

def customer_generator(env, bank):
    customer_id = 0
    while True:
        # Create new customers randomly between 1-5 minutes
        yield env.timeout(random.uniform(1, 5))
        customer_id += 1
        env.process(bank.customer_arrival(customer_id))

# Simulation setup
def run_bank_simulation(num_tellers=2, sim_time=100):
    env = simpy.Environment()
    bank = Bank(env, num_tellers)
    
    # Start customer generation process
    env.process(customer_generator(env, bank))
    
    # Run simulation
    env.run(until=sim_time)

# Run simulation with 2 tellers for 100 minutes
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    run_bank_simulation()