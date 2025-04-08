import os

# Metadata dosyasının yolu
metadata_file = "C:/Users/win11/Desktop/volodymyr labs(NN)/lab8_deneme/LJSpeech-1.1/metadata.csv"

# Dosya yolunu doğrulama
if not os.path.isfile(metadata_file):
    print("Hata: metadata.csv dosyası bulunamadı.")
    exit()

# Dosyayı açma ve içeriği okuma (utf-8 kodlaması kullanarak)
with open(metadata_file, "r", encoding="utf-8") as file:
    metadata = file.readlines()

# Veriyi işleme
data = []
for line in metadata:
    parts = line.strip().split("|")
    if len(parts) == 2:  # Beklenen format: dosya_adı | metin
        file_name, text = parts
        # Metni düşük harfe dönüştürme ve gereksiz karakterleri temizleme
        text = text.strip().lower()
        text = text.replace(",", "").replace(".", "").replace(";", "")
        data.append((file_name, text))

# Veriyi eğitim, doğrulama ve test kümelerine ayırma
train_data = data[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):int(len(data)*0.9)]
test_data = data[int(len(data)*0.9):]
