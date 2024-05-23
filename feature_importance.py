import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

processed_final = pd.read_csv("processed_final.csv")

y = processed_final['close']
X = processed_final.drop('close', axis=1).drop('date', axis=1).drop('tic', axis=1)
processed_final['Increase_Decrease'] = np.where(processed_final['volume'].shift(-1) > processed_final['volume'],1,0)
processed_final['Buy_Sell_on_Open'] = np.where(processed_final['open'].shift(-1) > processed_final['open'],1,0)
processed_final['Buy_Sell'] = np.where(processed_final['close'].shift(-1) > processed_final['close'],1,0)
processed_final['Returns'] = processed_final['close'].pct_change()
processed_final = processed_final.fillna(0)

# Create decision tree classifier object
clf = RandomForestRegressor(random_state=0, n_jobs=-1)

# Train model
model = clf.fit(X, y)

# Calculate feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Get the top 10 feature indices
top_10_indices = indices[:20]

# Rearrange feature names so they match the sorted feature importances
names = [processed_final.columns[i] for i in top_10_indices]

# Create plot
plt.figure(figsize=(10, 6))

# Create plot title
plt.title("Top 10 Feature Importance")

# Add bars
plt.bar(range(20), importances[top_10_indices])

# Add feature names as x-axis labels
plt.xticks(range(20), names, rotation=45, ha='right')

# Add labels and adjust layout
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()

# Show plot
plt.show()