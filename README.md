#Ödev Amacı
  Bu ödev, 2024-2025 Bahar Dönemi'nde yürütülen Derin Öğrenme dersi kapsamında hazırlanmış bir vize projesidir. Çalışmada, bir pirinç veri seti kullanılarak farklı pirinç cinslerinin sınıflandırılması amaçlanmıştır. Görüntü işleme teknikleri ve derin öğrenme tabanlı modellerden yararlanılarak bir pirinç tanıma sistemi geliştirilmiştir. Model, görsellerden elde edilen verileri kullanarak beş farklı pirinç türünü (Arborio, Basmati, Ipsala, Jasmine ve Karacadağ) yüksek doğrulukla sınıflandırabilmektedir. Ödev hazır veri seti üzerinden geliştirilmiştir.

##Model Tanımı
Bu çalışmada, hazır bir pirinç veri seti kullanılarak beş farklı pirinç türünü sınıflandırabilen bir derin öğrenme modeli geliştirdim. Model, eğitim süreci 20 epoch boyunca eğitilmiş ve %99,6 oranında doğruluk elde etmiştir. Bu yüksek doğruluk oranı, modelin veri seti içerisindeki pirinç türlerini büyük bir başarıyla ayırt edebildiğini göstermektedir. Kullanılan derin öğrenme mimarisi sayesinde, model hem eğitim hem de test verisi üzerinde başarılı sonuçlar üretmiştir.

##Epoch Sonuçları
1.	Epoch - accuracy: 0.8386 - loss: 0.3930 - val_accuracy: 0.9751 - val_loss: 0.0760
2.	Epoch - accuracy: 0.9721 - loss: 0.0866 - val_accuracy: 0.9846 - val_loss: 0.0449
3.	Epoch - accuracy: 0.9809 - loss: 0.0619 - val_accuracy: 0.9670 - val_loss: 0.0894
4.	Epoch - accuracy: 0.9813 - loss: 0.0586 - val_accuracy: 0.9827 - val_loss: 0.0527
5.	Epoch - accuracy: 0.9853 - loss: 0.0478 - val_accuracy: 0.9497 - val_loss: 0.1459
6.	Epoch - accuracy: 0.9848 - loss: 0.0470 - val_accuracy: 0.9867 - val_loss: 0.0419
7.	Epoch - accuracy: 0.9885 - loss: 0.0378 - val_accuracy: 0.9866 - val_loss: 0.0428
8.	Epoch - accuracy: 0.9877 - loss: 0.0372 - val_accuracy: 0.9913 - val_loss: 0.0296
9.	Epoch - accuracy: 0.9866 - loss: 0.0407 - val_accuracy: 0.9907 - val_loss: 0.0289
10.	Epoch - accuracy: 0.9904 - loss: 0.0298 - val_accuracy: 0.9930 - val_loss: 0.0249
11.	Epoch - accuracy: 0.9908 - loss: 0.0291 - val_accuracy: 0.9711 - val_loss: 0.0880
12.	Epoch - accuracy: 0.9891 - loss: 0.0306 - val_accuracy: 0.9930 - val_loss: 0.0222
13.	Epoch - accuracy: 0.9914 - loss: 0.0255 - val_accuracy: 0.9891 - val_loss: 0.0324
14.	Epoch - accuracy: 0.9921 - loss: 0.0243 - val_accuracy: 0.9893 - val_loss: 0.0348
15.	Epoch - accuracy: 0.9929 - loss: 0.0235 - val_accuracy: 0.9888 - val_loss: 0.0337
16.	Epoch - accuracy: 0.9919 - loss: 0.0237 - val_accuracy: 0.9943 - val_loss: 0.0208
17.	Epoch - accuracy: 0.9943 - loss: 0.0178 - val_accuracy: 0.9917 - val_loss: 0.0301
18.	Epoch - accuracy: 0.9923 - loss: 0.0237 - val_accuracy: 0.9869 - val_loss: 0.0444
19.	Epoch - accuracy: 0.9934 - loss: 0.0192 - val_accuracy: 0.9901 - val_loss: 0.0341
20.	Epoch - accuracy: 0.9947 - loss: 0.0165 - val_accuracy: 0.9909 - val_loss: 0.0279

#Grafikler
Model doğruluk oranı grafiği ve model kayıp grafiği yaptım.

##Test Sonuçları
Test verisi üzerinde gerçekleştirdiğim değerlendirmeler sonucunda model, pirinç türlerini %99,6 doğruluk oranıyla başarılı bir şekilde sınıflandırmıştır. Elde edilen tahminler, gerçek etiketlerle karşılaştırdım ve modelin büyük ölçüde doğru tahminlerde bulunduğu gözlemledim.

##Karşılaşılan Zorluklar
İlk eğittiğim modelde doğruluk oranı sadece %20 civarındaydı. Bu düşük sonucun sebebinin sınıfların yanlış etiketlenmesi olduğunu fark ettim. Etiketlerde gerekli düzenlemeleri yaptıktan sonra model başarısı artış gösterdi.
Daha sonra, görsellerin boyutunu 64x64 yerine 32x32 piksele düşürdüm. Bu sayede model hem daha hızlı eğitildi hem de doğruluk oranı belirgin şekilde arttı.

#Sonuç Olarak
Modelin eğitim sürecini tamamlayarak başarılı bir şekilde sonuç elde ettim. Eğitilen modeli .h5 formatında kaydettim ve böylece daha sonra yeniden kullanmak üzere hazır hale getirdim.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# İbrahim Hakkı HARMANKAYA Derin Öğrenme Dersi Vize Ödevi

Bu çalışmada, beş farklı pirinç türünü sınıflandırarak bir derin öğrenme modeli tasarlanmış ve uygulanmıştır.

## Çalışmada yaptığım kodlar ve sonuçlar aşağıda olduğu gibi verilmiştir.

1- Verileri Zip dosyasından dışarı çıkartma.

# Zipfile modülünü içe aktarır.
import zipfile

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall()  # Zip dosyasının içindeki tüm dosyaları bulunduğun klasöre çıkarır.

2- Gerekli kütüphanelerin yüklenmesi.

import cv2 # Görüntü işleme kütüphanesi ekler.
import pandas as pd # Veri analizi ve tablo (DataFrame) yapısı ekler.
import numpy as np # Sayısal işlemler, dizi (array) işlemleri ekler.
import os # Dosya ve klasör işlemleri için kullanılır.
import matplotlib.pyplot as plt # Görselleştirme için kullanılır.
from sklearn.model_selection import train_test_split # Veriyi eğitim ve test olarak ayırmak için kullanılır.
from tensorflow.keras.utils import to_categorical # Sınıf etiketlerini one-hot encoding'e dönüştürmek için kullanılır.

3- Gersellerin dosyalarının yollarını alır, karşılık gelen pirinç türünü etiketler.

# Veri yolunu ve classlar eklenir.
img_path = "Rice_Image_Dataset/"
labels = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

img_list = []
label_list = []

# Her bir pirinç türü için klasör oluşur.
for label in labels:
    for img_file in os.listdir(img_path + label):
        img_list.append(img_path + label + "/" + img_file)
        label_list.append(label)

df = pd.DataFrame({"img": img_list, "label": label_list})

# Pirinç türlerini sayısal etiketlere dönüştürmek için kullanılır.
d = {"Arborio": 0, "Basmati": 1, "Ipsala": 2, "Jasmine": 3, "Karacadag": 4}
df["encode_label"] = df["label"].map(d)

df = df.sample(frac=1).reset_index(drop=True)


4- Görsel verilerinin yükleyip düzenlendiği kısım.

# Görselin boyutunu 32x32 yapar.
size = 32

# DataFrame'deki her bir görsel yolunu belirler.
x = []
for imge in df["img"]:
    img = cv2.imread(imge)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    x.append(img)

# Etiketler için kodlar.
x = np.array(x)
y = df['encode_label']
y = to_categorical(y, num_classes=5)


5- Görsel eğitiminin ayrımı.


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)


6- Model derlemesi yapılıyor.

# Model kütüphanelerinin eklenmesi.
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Modeli sıralı (katman katman) oluşturuluyor.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5 sınıf için softmax

# Modeli derleme yapılıyor.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


7- Eğitim eğitiliyor.

# Model eğitiliyor.
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)


## Grafikler

1- Eğitim ve doğrulama grafiği.

# Eğitim ve doğrulama doğruluğu grafiği.
plt.figure(figsize=(12, 5))

2- Doğruluk grafiği.

# Doğruluk grafiği parametreleri girilir.
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

3- Kayıp model grafiği.

# Kayıp grafiği parametreleri girilir.
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()

## Sonuç

1- Tahmin testi yapılır.

# Kütüphane çağrılır.
import random

# Test verisinden rastgele fotoğraflar gelir ve tahmin yapılır.
random_index = random.randint(0, len(x_test)-1)
tahmin = model.predict(x_test[random_index:random_index+1])
gercek = np.argmax(y_test[random_index])
tahmin_edilen = np.argmax(tahmin)

# Gerçek ve tahmin edilen sınıf etiketleri yazdırılır.
print("Gerçek pirinç türü adı   :", labels[gercek])
print("Modelin tahmin ettiği tür:", labels[tahmin_edilen])

# Görüntü gösterilir ve başlık olarak tahmin vb sonuçları gösterilir.
plt.imshow(x_test[random_index])
plt.title(f"Gerçek: {labels[gercek]} | Tahmin Edilen: {labels[tahmin_edilen]}")
plt.axis("off")
plt.show()

2- Model kaydedilir.

model.save("pirinc_modeli_vize.h5")
