import numpy as np

from ordboost import OrdBoostClassifier, crps_score, pinball_loss

# Ordinal targets (e.g., pain scale 0-4)
X_train = np.random.randn(250, 5)
y_train = np.random.choice([0, 1, 2, 3, 4], size=250, p=[0.1, 0.2, 0.4, 0.2, 0.1])

model = OrdBoostClassifier(monotonicity="isotonic").fit(X_train, y_train)
dist = model.predict_dist(X_train)

# Calculate metrics
alpha = 0.75
crps = crps_score(y_train, dist)
p_loss_75 = pinball_loss(y_train, dist.ppf(alpha), q=alpha)

print(f"Discrete CRPS: {crps:.4f}")
print(f"Pinball Loss (75th percentile): {p_loss_75:.4f}")
