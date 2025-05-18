# Import semua library yang diperlukan
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_pickle, save_pickle
import os

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'train_test_data.pkl')
X_train, X_test, y_train, y_test = load_pickle(model_path)

# Latih model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.savefig('../outputs/confusion_matrix.png')
plt.close()