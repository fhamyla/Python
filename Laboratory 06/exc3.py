import random

def estimate_probability(trials): 
    count = 0 
    for _ in range(trials): 
        die1 = random.randint(1, 6) 
        die2 = random.randint(1, 6) 
        if die1 + die2 == 7: 
            count += 1 
    return count / trials 

probability = estimate_probability (100000) 
print(f"Estimated probability of sum = 7 when rolling two dice: {probability}")