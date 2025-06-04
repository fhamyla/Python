import random

def estimate_pi(n_trials):
    inside_circle = 0
    for _ in range(n_trials):
        x, y = random.uniform(0, 1), random.uniform(0, 1)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4 * inside_circle / n_trials

for num_trials in [1000, 10000, 100000]:
    pi_estimate = estimate_pi(num_trials)
    print(f"Estimated value of pi after {num_trials} trials: {pi_estimate}")