# 📘 Whisperer: Real-Time Speech Synchronization & Intelligent Teleprompter

📌 *Whisperer*, klasik prompter sistemlerinin konuşmacıyı senaryoya bağımlı kılan yapısının aksine, gerçek zamanlı konuşma tanıma ile senaryo eşleşmesini esnek, dinamik ve doğal bir şekilde gerçekleştiren; senkronize ve anlamsal olarak toleranslı bir sistem sunar.

Whisperer, konuşmacı ile metin arasındaki senkronizasyonu gerçek zamanlı takip eden, kullanıcıya görsel ve sesli destek sağlayan akıllı bir teleprompter uygulamasıdır. Uygulama; *Google Speech-to-Text (STT), Text-to-Speech (TTS), Deepgram, FastText, Sentence-BERT* gibi modern NLP araçlarıyla donatılmıştır.

---

## 🧩 Kullandığımız Teknolojiler ve Sistem Bileşenleri

### 🔊 Dinamik Konuşma Sentezi (TTS)

Google Cloud Text-to-Speech servisini kullanan sistemimiz, ses çıktısını kullanıcının konuşma hızına (WPS - kelime/saniye) göre doğrusal dönüşüm algoritmalarıyla ayarlayarak konuşma ritmine ve duraklamalara uyumlu hale getirir.

---

### 🗣 Gerçek Zamanlı Konuşma Tanıma (STT)

Google Cloud Speech-to-Text ile entegre edilen Whisperer, mikrofon verilerini *PyAudio* ve *Sounddevice* kütüphaneleri aracılığıyla yakalar ve düşük gecikmeli streaming modunda işleyerek yüksek doğruluklu, anlık metin dönüşümü sağlar.

---

### 🧠 Doğal Dil İşleme ile Semantik Eşleştirme (NLP)

- *Homofon algılama* için *CMUDict* fonetik sözlüğü kullanılır.  
- *Eş anlamlı kelime analizi* için *FastText* modeli tercih edilir.  
- *Cümle düzeyinde anlam eşleşmeleri* için *BERT (MPNet-base-v2)* modeli, cosine similarity yöntemiyle yüksek doğrulukta semantik eşleşme sağlar.

---

### 🧩 Chunking ve Senkronizasyon

- Metinler, *Miller’ın 7±2 kuralı* ve *bilişsel yük* prensiplerine göre yaklaşık 40 karakterlik *chunk*’lara (anlamlı parçalara) bölünür.
- Chunk'lar, kullanıcı konuşma ritmine göre otomatik olarak senkronize edilir.
- Geçişler sırasında *BERT tabanlı* ve *kural tabanlı* metin benzerliği algoritmaları kullanılır.

---

### 🖥 Arayüz Tasarımı

Arayüz, *CustomTkinter* ile geliştirilmiştir.  
Anlık görsel geri bildirim, vurgulama özellikleri ve kullanıcı tarafından düzenlenebilir metin parçalama seçenekleri sunar.

---

### 📊 Renkli Geri Bildirim Sistemi

Söylenen kelimeler:

- ✅ *Yeşil* → Tam eşleşme  
- 🔶 *Turuncu* → Eşsesli  
- 🔵 *Mavi* → Anlamsal benzerlik  
- ⚪ *Beyaz* → Eşleşmeyen kelime  

---

### 🎵 Kişiselleştirilmiş Ritim Kalibrasyonu

*Deepgram* ile kullanıcıların konuşma kayıtları analiz edilerek ortalama konuşma hızı ve duraklama süreleri hesaplanır. Sistem, *TTS parametrelerini* bu verilere göre otomatik olarak kalibre eder.

---

## 🛠 Kullanılan Teknolojiler

- Python  
- CustomTkinter  
- Google Cloud Speech-to-Text & Text-to-Speech  
- Deepgram API  
- Sentence-BERT (paraphrase-mpnet-base-v2)  
- FastText (wiki.simple.vec)  
- CMU Pronouncing Dictionary  
- NumPy, SoundDevice, PyAudio, Gensim, NLTK, Requests  

---

## 📦 Kurulum

### 1. Gerekli Paketleri Yükleyin

bash
pip install -r requirements.txt


---

### 2. Aşağıdaki Dosyaları Klasörünüze Ekleyin

#### 📥 Gerekli Vektör Dosyalarını İndirin

Uygulama, kelime düzeyinde anlamsal benzerlikleri hesaplamak için *FastText* model dosyalarını kullanmaktadır. Bu dosyalar depoya yüklenmemiştir çünkü boyutları çok büyüktür.

#### Gerekli dosyalar:

- *cc.en.300.vec*  
  https://fasttext.cc/docs/en/crawl-vectors.html adresine gidin ve “English” başlığının altındaki "text" bağlantısına tıklayarak *cc.en.300.vec.gz* dosyasını indirin.

- *wiki.simple.vec*  
  Bu dosyayı aşağıdaki adresten indirin:  
  https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec

> Her iki dosyayı da Whisperer_project.py ile aynı klasöre koyduğunuzdan emin olun.

---

### 3. Google Cloud API Ayarları (STT ve TTS için)

1. Google Cloud Console üzerinden bir proje oluşturun.  
2. Speech-to-Text ve Text-to-Speech API’lerini etkinleştirin.  
3. Bir *Service Account* oluşturun ve JSON kimlik dosyasını indirin.  
4. Bu dosyayı proje klasörüne yerleştirin.  
5. Whisperer_project.py içindeki şu satırı kendi dosya adınıza göre düzenleyin:

python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_GOOGLE_CREDENTIALS_JSON_PATH.json"


---

### 4. Deepgram API Ayarları

1. https://developers.deepgram.com/ üzerinden hesap oluşturun.  
2. API anahtarınızı alın.  
3. app.py içinde şu satırı kendi anahtarınızla değiştirin:

python
DEEPGRAM_API_KEY = "YOUR_DEEPGRAM_API_KEY"
![Uygulama Ekran Görüntüsü](images.jpg)
