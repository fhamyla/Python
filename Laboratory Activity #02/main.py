import simpy
import random
import numpy

NO_OF_CUSTOMERS = 1000
SIMULATION_RUNS = 10
OPENING_HOURS = 15 * 60
INTERARRIVAL_TIME = OPENING_HOURS / NO_OF_CUSTOMERS

menu = {
    1: ["Regular Coffee", 10, 15],
    2: ["Latte", 30, 45],
    3: ["Mocha", 30, 45],
    4: ["Cold Brew", 10, 20],
    5: ["Frappe", 50, 70],
    6: ["Espresso", 20, 35],
    7: ["Coffee of the Day", 60, 80]
}

payment_options = {
    1: ["Cash", 15, 30],
    2: ["Card", 10, 20],
    3: ["Online Payment", 5, 15]
}

def run_simulation(no_cashiers, no_baristas):
    env = simpy.Environment()
    cashier = simpy.Resource(env, no_cashiers)
    barista = simpy.Resource(env, no_baristas)

    order_wait_time = []
    order_time = []

    def customer(env, name):
        arrival_time = env.now
        
        with cashier.request() as req:
            yield req
            menu_item = random.randint(1, 7)
            payment_type = random.randint(1, 3)
            time_to_order = random.randint(payment_options[payment_type][1], payment_options[payment_type][2])
            
            yield env.timeout(time_to_order)
        
        with barista.request() as req:
            start_bq = env.now
            yield req
            order_wait_time.append(env.now - start_bq)
            
            time_to_prepare = random.randint(menu[menu_item][1], menu[menu_item][2])
            yield env.timeout(time_to_prepare)
            order_time.append(env.now - arrival_time)
    
    def generate_customers(env):
        for i in range(NO_OF_CUSTOMERS):
            yield env.timeout(random.uniform(0.3 * INTERARRIVAL_TIME, 1.2 * INTERARRIVAL_TIME))
            env.process(customer(env, i))
    
    env.process(generate_customers(env))
    env.run(until=OPENING_HOURS)
    
    avg_service_time = numpy.mean(order_time)
    return {
        "order_wait": numpy.mean(order_wait_time) if order_wait_time else 0,
        "order_time": avg_service_time,
        "valid": avg_service_time < 120
    }

optimal_cashiers, optimal_baristas = 2, 4
while True:
    results = [run_simulation(optimal_cashiers, optimal_baristas) for _ in range(SIMULATION_RUNS)]
    avg_order_time = numpy.mean([r["order_time"] for r in results])
    
    if avg_order_time < 120:
        break
    optimal_cashiers += 1
    optimal_baristas += 1

avg_order_wait = numpy.mean([r["order_wait"] for r in results])
avg_order_time = numpy.mean([r["order_time"] for r in results])

print("\nSimulation Results")
print(f"Optimal Cashiers: {optimal_cashiers}, Optimal Baristas: {optimal_baristas}")
print(f"Average wait time in order queue: {avg_order_wait:.2f} seconds")
print(f"Average time until order is serviced: {avg_order_time:.2f} seconds")