import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import save_pickle

# Load data
data = pd.read_csv('../data/diabetes.csv')

# Pisahkan fitur dan target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Simpan scaler dan data
save_pickle(scaler, '../models/scaler.pkl')
save_pickle((X_train, X_test, y_train, y_test), '../models/train_test_data.pkl')

print("Preprocessing selesai! Data tersimpan di /models")