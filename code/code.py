import matplotlib.pyplot as plt

birds = {
    "Pigeon": {"weight_kg": 0.3, "wingspan_m": 0.64},
    "Seagull": {"weight_kg": 1.0, "wingspan_m": 1.3},
    "Eagle": {"weight_kg": 6.0, "wingspan_m": 2.3},
    "Albatross": {"weight_kg": 9.0, "wingspan_m": 3.5},
    "Condor": {"weight_kg": 11.0, "wingspan_m": 3.2},
    "Hypothetical Human": {"weight_kg": 75, "wingspan_m": 12}
}

names = list(birds.keys())
weights = [birds[bird]["weight_kg"] for bird in names]
wingspans = [birds[bird]["wingspan_m"] for bird in names]

plt.figure(figsize=(10, 6))
plt.scatter(weights[:-1], wingspans[:-1], color='blue', label='Birds', s=100)
plt.scatter(weights[-1], wingspans[-1], color='red', label='Hypothetical Human', s=100)
plt.title("Wingspan vs. Weight for Birds and a Hypothetical Flying Human")
plt.xlabel("Weight (kg)")
plt.ylabel("Wingspan (m)")
plt.grid(True)
plt.legend()

for i, name in enumerate(names):
    plt.annotate(name, (weights[i] + 0.5, wingspans[i]))

plt.tight_layout()
plt.show()