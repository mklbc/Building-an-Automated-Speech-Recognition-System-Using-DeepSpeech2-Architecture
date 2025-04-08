import matplotlib.pyplot as plt
import numpy as np

# Eğitim ve doğrulama kayıplarını görselleştirme
history = ...  # Modelin eğitim geçmişi
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Rastgele birkaç örneği inceleme
predictions = ...  # Modelin tahminleri
targets = ...      # Gerçek etiketler
for i in np.random.randint(0, len(predictions), 2):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 100)
