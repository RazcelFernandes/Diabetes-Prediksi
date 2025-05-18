from utils import load_pickle
import numpy as np

# Load model dan scaler
model = load_pickle('../models/diabetes_model.pkl')
scaler = load_pickle('../models/scaler.pkl')

# Contoh data baru (Pregnancies, Glucose, ..., Age)
new_data = np.array([[2, 120, 70, 20, 80, 25.0, 0.3, 30]])  # Sesuaikan!

# Normalisasi
new_data_scaled = scaler.transform(new_data)

# Prediksi
prediction = model.predict(new_data_scaled)
print("Hasil Prediksi:", "Diabetes" if prediction[0] == 1 else "Tidak Diabetes")