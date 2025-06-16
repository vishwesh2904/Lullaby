import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import runpy

# Load the dataset
df = pd.read_csv("data/insomnia_synthetic.csv")

# Basic info and statistics
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check class distribution
print("\nInsomnia Level Distribution:")
print(df["Insomnia Level"].value_counts())

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Insomnia Level", order=df["Insomnia Level"].value_counts().index)
plt.title("Insomnia Level Distribution")
plt.savefig("insomnia_level_distribution.png")
plt.close()

# Plot correlation heatmap of features
plt.figure(figsize=(12, 10))
corr = df.drop(columns=["Insomnia Level"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("feature_correlation_heatmap.png")
plt.close()

print("Plots saved: insomnia_level_distribution.png, feature_correlation_heatmap.png")


