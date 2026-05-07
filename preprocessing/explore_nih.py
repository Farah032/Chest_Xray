import pandas
import matplotlib.pyplot as plt
import os

base_path = "/Users/farahjabeen/Desktop/XRAY_PROJECT/data/Data_Entry_2017.csv"

df  = pandas.read_csv(base_path)
'''
print('\n shape:',df.shape)
print('\nColumns Name:',df.columns)
print('\nrows head',df.head(5))
print('\n Unique values in label column:',df["Finding Labels"].nunique())
print('\n Unique values in label column:',df["Finding Labels"].value_counts())
'''

# Split labels and create one label per row
all_labels = df['Finding Labels'].str.split('|').explode()

# Count frequency of each disease
label_counts = all_labels.value_counts()

# Print counts
print("\nDisease Counts:\n")
print(label_counts)

# Count No Finding
no_finding_count = label_counts["No Finding"]

# Percentage of No Finding
total_images = len(df)
no_finding_percentage = (no_finding_count / total_images) * 100

print("\nNo Finding Count:", no_finding_count)
print("No Finding Percentage: {:.2f}%".format(no_finding_percentage))

# Sort ascending for horizontal bar plot
# (largest appears at top after invert_yaxis)
label_counts_sorted = label_counts.sort_values(ascending=True)

# Create figure
plt.figure(figsize=(10, 8))

# Horizontal bar chart
plt.barh(label_counts_sorted.index, label_counts_sorted.values)

# Labels and title
plt.xlabel("Count of Images")
plt.ylabel("Disease Name")
plt.title("NIH ChestX-ray14 — Disease Distribution (n=112,120)")

# Put most frequent at top
plt.gca().invert_yaxis()

# Prevent label cutoff
plt.tight_layout()

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# Save figure
plt.savefig("results/nih_disease_distribution.png")

print("\nPlot saved to results/nih_disease_distribution.png")

# Optional: display plot
plt.show()