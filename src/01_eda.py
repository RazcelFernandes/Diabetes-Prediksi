import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
data = pd.read_csv('../data/diabetes.csv')

# EDA dasar
print("Info Dataset:")
print(data.info())
print("\nStatistik Deskriptif:")
print(data.describe())

# Visualisasi
os.makedirs('../outputs', exist_ok=True)  # Buat folder outputs

# Distribusi Glucose
plt.figure(figsize=(8, 4))
sns.histplot(data['Glucose'], kde=True, color='blue')
plt.title('Distribusi Kadar Glukosa')
plt.savefig('../outputs/glucose_dist.png')  # Simpan gambar
plt.close()

# Heatmap korelasi
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Korelasi Antar Fitur')
plt.savefig('../outputs/correlation_heatmap.png')
plt.close()