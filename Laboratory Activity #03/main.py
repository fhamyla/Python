import numpy as np
import matplotlib.pyplot as plt

# Step 2: Generate customer arrival times using an exponential distribution
num_customers = 20
arrival_times = np.cumsum(np.random.exponential(scale=2, size=num_customers))  

# Step 3: Generate service times randomly (1 to 3 minutes)
service_times = np.random.randint(1, 4, size=num_customers)

# Step 4: Compute waiting and departure times with two servers
waiting_times = np.zeros(num_customers)
departure_times = np.zeros(num_customers)
server_busy = [0, 0]  # Two servers, tracking when each is free

for i in range(num_customers):
    chosen_server = 0 if server_busy[0] <= server_busy[1] else 1  # Choose the less busy server
    waiting_times[i] = max(0, server_busy[chosen_server] - arrival_times[i])
    departure_times[i] = arrival_times[i] + waiting_times[i] + service_times[i]
    server_busy[chosen_server] = departure_times[i]  # Update the chosen server's availability

# Step 5: Plot customer wait times
plt.plot(range(num_customers), waiting_times, marker='o', linestyle='-', color='b')
plt.xlabel('Customer Number')
plt.ylabel('Waiting Time (minutes)')
plt.title('Queue System Waiting Times (Two Servers)')
plt.grid(True)
plt.show()