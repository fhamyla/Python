import random
import matplotlib.pyplot as plt 

def plot_points (n): 
    x_in, y_in, x_out, y_out = [], [], [], []
    for _ in range(n): 
        x, y = random.random(), random.random() 
        if x**2 + y**2 <= 1: 
            x_in.append(x) 
            y_in.append(y) 
        else: 
            x_out.append(x) 
            y_out.append(y) 
            
    plt.figure(figsize=(6,6)) 
    plt.scatter(x_in, y_in, color='green', s=1, label='Inside Circle') 
    plt.scatter(x_out, y_out, color='red', s=1, label='Outside Circle') 
    plt.title('Monte Carlo Estimation of π') 
    plt.legend() 
    plt.show() 

plot_points (10000)