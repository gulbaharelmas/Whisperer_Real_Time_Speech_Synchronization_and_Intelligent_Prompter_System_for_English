
import os
import threading
import time
import numpy as np
import sounddevice as sd
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext
import google.cloud.speech as speech
import google.cloud.texttospeech as tts
import pyaudio
import re
from difflib import SequenceMatcher
from collections import defaultdict
import nltk
from nltk.corpus import cmudict
from gensim.models import KeyedVectors
from word2number import w2n
from tkinter import simpledialog
import requests
from sentence_transformers import SentenceTransformer, util
import customtkinter as ctk
from customtkinter import CTkTextbox

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_GOOGLE_CREDENTIALS_JSON_PATH.json"  # Google Cloud kimlik doğrulama dosyasının yolunu girin


bert_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')  
import wave

DEEPGRAM_API_KEY = "YOUR_DEEPGRAM_API_KEY"  # Deepgram API anahtarınızı buraya girin

user_rhythm = {"speaking_rate": 1.0, "pause_ms": 300}

class EmbeddedWaveformVisualizer:
    def __init__(self, parent_frame, width=280, height=80):
        self.canvas_width = width
        self.canvas_height = height
        self.canvas = tk.Canvas(parent_frame, width=self.canvas_width, height=self.canvas_height, bg='#2a292f',  highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(pady=5)

        self.volume_history = [0] * 100
        self.running = False
        self.amplify = 300

        self.stream = sd.InputStream(
            channels=1,
            samplerate=44100,
            blocksize=1024,
            callback=self.audio_callback
        )

    def audio_callback(self, indata, frames, time, status):
        amplitude = np.mean(np.abs(indata))
        scaled_volume = np.clip(amplitude * self.amplify, 0, self.canvas_height // 2)
        self.volume_history.append(scaled_volume)
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)

    def start(self):
        if not self.running:
            self.running = True
            self.stream.start()
            self.update_canvas()

    def stop(self):
        if self.running:
            self.running = False
            self.stream.stop()

    def update_canvas(self):
        self.canvas.delete("all")
        center_y = self.canvas_height // 2
        spacing = self.canvas_width / len(self.volume_history)
        line_width = 2

        for i, volume in enumerate(self.volume_history):
            x = i * spacing
            y0 = center_y - volume
            y1 = center_y + volume
            self.canvas.create_line(x, y0, x, y1, fill="#6661ea", width=line_width, capstyle=tk.ROUND)

        if self.running:
            self.canvas.after(30, self.update_canvas)



def update_chunk_editor():
    chunk_editor.delete(1.0, tk.END)
    if changed_chunks:
        chunk_editor.insert(tk.END, "\n".join(changed_chunks))
    else:
        chunk_editor.insert(tk.END, "\n".join(chunk_by_character_limit(original_text, max_chars=40)))

def bert_similarity(sent1, sent2, threshold=0.70):
    if not sent1.strip() or not sent2.strip():
        return 0.0, False  
    embeddings = bert_model.encode([sent1, sent2], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return score, score >= threshold

def wps_to_speaking_rate(wps):
    min_wps, max_wps = 2.19, 4.0
    min_rate, max_rate = 0.65, 1.25

    # Aşırı değerleri sınırlara getirir
    wps = max(min(wps, max_wps), min_wps)

    # Doğrusal dönüşüm
    speaking_rate = ((wps - min_wps) / (max_wps - min_wps)) * (max_rate - min_rate) + min_rate
    return round(speaking_rate, 2)





def record_sample_until_silence(filename="calibration.wav", samplerate=16000, silence_threshold=100, silence_duration=1.5):

    buffer = []
    chunk_duration = 0.1
    chunk_size = int(samplerate * chunk_duration)
    silence_chunks = int(silence_duration / chunk_duration)
    silence_buffer = []

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype='int16')
    stream.start()

    try:
        while True:
            audio_chunk, _ = stream.read(chunk_size)
            audio_data = np.abs(audio_chunk)
            volume = np.mean(audio_data)

            if volume < silence_threshold:
                silence_buffer.append(audio_chunk)
                if len(silence_buffer) >= silence_chunks:
                    print("🟢 Sessizlik algılandı. Kayıt tamamlandı.")
                    break
            else:
                buffer.extend(silence_buffer)
                silence_buffer.clear()
                buffer.append(audio_chunk)
    finally:
        stream.stop()
        stream.close()

    recording = np.concatenate(buffer)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

def analyze_with_deepgram(audio_path):
    with open(audio_path, 'rb') as audio:
        response = requests.post(
            'https://api.deepgram.com/v1/listen',
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type': 'audio/wav'
            },
            data=audio
        )
    print("Deepgram HTTP Status:", response.status_code)
    print("Deepgram Response Preview:", response.json())
    return response.json()

def extract_rhythm_from_deepgram(json_data):
    try:
        word_timings = json_data["results"]["channels"][0]["alternatives"][0]["words"]
    except:
        print("❌ Deepgram çıktısı geçersiz.")
        return {"speaking_rate": 1.0, "pause_ms": 300}

    if len(word_timings) < 2:
        return {"speaking_rate": 1.0, "pause_ms": 300}

    pauses, durations = [], []

    for i in range(1, len(word_timings)):
        prev, curr = word_timings[i - 1], word_timings[i]
        pauses.append(curr['start'] - prev['end'])
        durations.append(curr['end'] - curr['start'])

    avg_pause = sum(pauses) / len(pauses)
    avg_duration = sum(durations) / len(durations)
    wps = 1 / avg_duration

    speaking_rate = wps_to_speaking_rate(wps)

    pause_ms = int(avg_pause * 1000)
    pause_ms = max(300, min(pause_ms, 1000))

    print(f"[DEEPGRAM] WPS: {wps:.2f}, Pause: {pause_ms}ms → speaking_rate: {speaking_rate:.2f}")

    return {"speaking_rate": speaking_rate, "pause_ms": pause_ms}

# --- Google STT (Speech-to-Text) Ayarları --- #
stt_client = speech.SpeechClient()
recognition_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
)
streaming_config = speech.StreamingRecognitionConfig(
    config=recognition_config, 
    interim_results=True,
    single_utterance=False  # Sürekli tanıma için kullanılır
)

# --- Google TTS (Text-to-Speech) Ayarları --- #
tts_client = tts.TextToSpeechClient()

# --- PyAudio Ayarları --- #
p = pyaudio.PyAudio()
audio_queue = queue.Queue(maxsize=10)

def audio_callback(in_data, frame_count, time_info, status):
    # Mikrofon verilerini kuyruğa ekler
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024,
    stream_callback=audio_callback,
)

stream.start_stream()

CONJUNCTIONS = {"and", "but", "or", "so", "yet"}

def test_rhythm_playback():
    rate_value = user_rhythm["speaking_rate"]  

    print(f"[TEST RYTHM] Using speaking_rate={rate_value:.2f} / pause={user_rhythm['pause_ms']}ms")
    print(f"[TEST] Final speaking_rate used in Google TTS: {rate_value:.2f}")

    test_sentence = "This is a test of your current speaking rhythm."

    ssml = create_ssml(test_sentence, pause_ms=user_rhythm["pause_ms"])
    synthesis_input = tts.SynthesisInput(ssml=ssml)

    voice = tts.VoiceSelectionParams(language_code="en-US", ssml_gender=tts.SsmlVoiceGender.NEUTRAL)
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        speaking_rate=rate_value
    )

    try:
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        wav = np.frombuffer(response.audio_content, dtype=np.int16)
        sd.play(wav, samplerate=24000)
        sd.wait()
    except Exception as e:
        print(f"❌ Test playback error: {e}")

def chunk_by_character_limit(text, max_chars=40):
    words = text.split()
    chunks = []
    current_chunk = []

    for i, word in enumerate(words):
        clean_word = word.strip(",.!?").lower()
        is_end_punctuation = any(punc in word for punc in [".", "!", "?"])
        has_comma = "," in word
        is_conjunction = clean_word in CONJUNCTIONS

        if has_comma:
            current_chunk.append(word)
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        elif is_end_punctuation:
            current_chunk.append(word)
            if len(current_chunk) >= 6 or is_end_punctuation:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        elif is_conjunction:
            if len(current_chunk) >= 6:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
        else:
            if len(" ".join(current_chunk)) + len(word) + 1 <= max_chars:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --- Metin İşleme Fonksiyonları --- #
def clean_text(text):
    # Noktalama işaretlerini kaldırır ve küçük harfe çevirir
    return re.sub(r'[^\w\s]', '', text.lower())


def are_numerically_equivalent(w1, w2):
    try:
        # "100" -> 100, "one hundred" -> 100 numara dönüşümü yapar
        n1 = w2n.word_to_num(w1) if not w1.isdigit() else int(w1)
        n2 = w2n.word_to_num(w2) if not w2.isdigit() else int(w2)
        return n1 == n2
    except:
        return False


nltk.download('cmudict')  # CMU sözlüğünü indirir
pron_dict = cmudict.dict()

def are_homophones(w1, w2):
    w1, w2 = w1.lower(), w2.lower()
    try:
        return any(p1 == p2 for p1 in pron_dict[w1] for p2 in pron_dict[w2])
    except:
        return False

# --- FastText Modeli ve Benzerlik --- #
from gensim.models import KeyedVectors
print("🔁 FastText modeli yükleniyor...")
model = KeyedVectors.load_word2vec_format("wiki.simple.vec", binary=False)
print("✅ FastText modeli yüklendi.")

def fasttext_similarity(w1, w2):
    try:
        return model.similarity(w1.lower(), w2.lower())
    except KeyError:
        return 0.0

def calculate_similarity_detailed(ref_text, stt_text):
    ref_words = clean_text(ref_text).split()
    stt_words = clean_text(stt_text).split()

    matched = 0
    ref_match_counts = defaultdict(int)
    ref_total_counts = {word: ref_words.count(word) for word in ref_words}
    
    match_info = {}  # Her kelime için eşleşme türü

    for ref_word in ref_words:
        match_info[ref_word] = "none"

    for stt_word in stt_words:
        best_score = 0
        best_match = None
        match_type = "none"

        for ref_word in ref_words:
            if match_info[ref_word] == "exact":
                continue  

            if stt_word == ref_word:
                best_score = 1.0
                best_match = ref_word
                match_type = "exact"
                break
            elif are_numerically_equivalent(stt_word, ref_word):
                if best_score < 1.0:
                    best_score = 1.0
                    best_match = ref_word
                    match_type = "numeric"


            elif are_homophones(stt_word, ref_word):
                if best_score < 0.8:
                    best_score = 1.0
                    best_match = ref_word
                    match_type = "homophone"
            else:
                score = fasttext_similarity(stt_word, ref_word)
                if score > best_score and score >= 0.4:
                    best_score = 1.0
                    best_match = ref_word
                    match_type = "semantic"

        if best_match:
            if ref_match_counts[best_match] < ref_total_counts.get(best_match, 1):
                # Yalnızca izin verilen sayıda eşleşme varsa puan verir
                if match_type in {"exact", "numeric", "homophone", "semantic"}:
                    matched += 1.0
                else:
                    matched += 0.7
                ref_match_counts[best_match] += 1
                match_info[best_match] = match_type
            else:
                # Eşleşme sınırı aşıldıysa sadece match_info güncellenir (skor verilmez)
                match_info[best_match] = match_type

    similarity_percent = round((matched / len(ref_words)) * 100, 2) if ref_words else 0
    return similarity_percent, match_info


# --- Örnek Metin --- #
text = "Hello. Welcome to the Whisperer Project."# --- Original Text ve Changed Chunks --- #
original_text = text  # Başlangıçta örnek metinle başlatır
changed_chunks = []   # Kullanıcı 2. sayfada chunkları değiştirirse burada tutulur

# Metni cümlelere ayırır
sentences = chunk_by_character_limit(text, max_chars=40)

# Değişkenleri başlatır
current_sentence_index = 0
speech_ready = threading.Event()
recognized_text = ""
remain_text = ""
recognized_words_set = set()
sentence_completed = False
similarity_threshold = 80 # %70 benzerlik eşiği
recognition_running = False

# --- Sessizlik ve dolgu ifadeleri için değişkenler --- #
last_speech_time = time.time()
silence_threshold = 3  # Sessizlik süresi eşiği (saniye)
tts_speaking = False
tts_end_time = 0
progress_pause = False  # Cümle ilerlemesi için duraklatır
silence_timer_active = False  # Silence timer sadece start'a basılınca çalışır
recognize_thread_ref = None
speak_thread_ref = None



fallback_phrases = [
    "moving forward",
    "continuing",
    "in fact",
    "next point",
    "go on",
    "move on",
    "next sentence",
    "skip this",
    "i'll proceed"
]

# --- Tkinter Arayüzü --- #
root = ctk.CTk()
root.title("Smart Teleprompter")
root.geometry("1000x600")
root.configure(fg_color="#121117")



# İlk cümleyi gösterir
current_sentence = sentences[current_sentence_index]
sentence_words = current_sentence.split()
word_labels = []


def create_ssml(text, pause_ms=300):
    return f'<speak>{text}<break time="{pause_ms}ms"/></speak>'


def update_word_display():
    # Kelime etiketlerini günceller
    global word_labels, sentence_words
    
    # Önceki kelime etiketlerini temizler
    for label in word_labels:
        label.destroy()
    word_labels.clear()
    
    # Yeni kelime etiketleri oluştur
    for word in sentence_words:
        clean_word = clean_text(word)
        color = "white"  
        if clean_word in recognized_words_set:
            color = "green"
        
        label = tk.Label(word_frame, text=word + " ", font=("Calibri", 20, "bold"), fg=color, bg="#2a292f")
        label.pack(side=tk.LEFT)
        word_labels.append(label)


def update_fallback_phrases():
    # Dolgu ifadelerini metin kutusundan günceller
    global fallback_phrases
    text = fallback_text.get(1.0, tk.END).strip()
    fallback_phrases = [p.strip().lower() for p in text.split("\n") if p.strip()]
    if not fallback_phrases:  # Boşsa varsayılan ifadeleri kullanır
        fallback_phrases = ["let me continue"]

def update_silence_timer():
    global recognition_running, silence_timer_active

    if not silence_timer_active:
        return  # Silence timer aktif değilse hiç çalıştırır

    if not recognition_running:
        return  # Tanıma kapandıysa çalıştırır

    global last_speech_time, progress_pause, tts_speaking, tts_end_time

    if tts_speaking and time.time() < tts_end_time:
        silence_duration = 0
    else:
        silence_duration = time.time() - max(last_speech_time, tts_end_time)

    if silence_duration < 0:
        silence_duration = 0

    silence_label.configure(text=f"Silence: {silence_duration:.1f}s")

    if silence_duration > silence_threshold and recognition_running and not progress_pause:
        status_label.configure(text=f"Long silence detected ({silence_duration:.1f}s). Say a fallback phrase to continue.")

    root.after(200, update_silence_timer)


def update_ui(recognized_text, similarity, bert_score=0.0, remain=""):
    recognized_text_area.delete(1.0, tk.END)
    recognized_text_area.insert(tk.END, recognized_text.lower())
    similarity_label.configure(text=f"Similarity: {similarity:.2f}%")
    bert_label.configure(text=f"BERT: {bert_score:.2f}%")
    #remain_text_area.delete(1.0, tk.END)
    #remain_text_area.insert(tk.END, remain.strip())

    if current_sentence_index < len(sentences):
        sentence_label.configure(text=f"Current Sentence ({current_sentence_index + 1}/{len(sentences)}): {sentences[current_sentence_index]}")
    
    root.update()


def update_word_highlighting(match_info):
    global word_labels, sentence_words
    
    for i, word in enumerate(sentence_words):
        clean_word = clean_text(word)
        match_type = match_info.get(clean_word, "none")
        
        if match_type == "exact":
            word_labels[i].config(fg="green")
        elif match_type == "homophone":
            word_labels[i].config(fg="orange")
        elif match_type == "semantic":
            word_labels[i].config(fg="blue")
        elif match_type == "numeric":
            word_labels[i].config(fg="green")
        else:
            word_labels[i].config(fg="white")
        


def check_for_fallback_phrase(text):
    clean_recognized = clean_text(text.lower())
    
    for phrase in fallback_phrases:
        if phrase.lower() in clean_recognized:
            return True
    return False

def move_to_next_sentence():
    # Bir sonraki cümleye geçer
    global current_sentence_index, recognized_words_set, progress_pause
    
    status_label.configure(text=f"Moving to next sentence...")
    current_sentence_index += 1
    
    if current_sentence_index < len(sentences):
        # Sonraki cümle için günceller
        current_sentence = sentences[current_sentence_index]
        current_sentence_words = current_sentence.split()
        sentence_words.clear()
        sentence_words.extend(current_sentence_words)
        recognized_words_set.clear()
        update_word_display()
        
        # Cümle ilerlemesini kısa süre duraklatır (hatalı tetiklemeleri önlemek için)
        progress_pause = True
        root.after(2000, lambda: setattr(globals()['progress_pause'], 'value', False))
        
        # Sonraki cümle için hazır olduğumuzu bildirir
        speech_ready.set()
        return True
    else:
        status_label.configure(text="All sentences completed!")
        recognition_running = False
        speech_ready.set()
        start_button.configure(state=tk.NORMAL)
        return False

def recognize_speech():
    global current_sentence_index, recognized_text, word_labels, sentence_completed, recognized_words_set
    global recognition_running, last_speech_time, sentence_words, progress_pause

    recognition_running = True

    while current_sentence_index < len(sentences) and recognition_running:
        status_label.configure(text=f"Listening for sentence {current_sentence_index + 1}/{len(sentences)}...")

        recognized_text = ""
        cumulative_text = ""
        remain_text = ""
        recognized_words_set.clear()
        sentence_completed = False
        last_speech_time = time.time()
        progress_pause = False

        update_ui(recognized_text, 0, 0, remain_text)

        def audio_generator():
            while recognition_running:
                try:
                    chunk = audio_queue.get(timeout=0.5)
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    continue

        try:
            responses = stt_client.streaming_recognize(streaming_config, audio_generator())

            for response in responses:
                if not recognition_running:
                    break

                if len(response.results) == 0:
                    continue

                result = response.results[0]
                current_transcript = result.alternatives[0].transcript.strip()
                recognized_text = current_transcript

                if check_for_fallback_phrase(recognized_text):
                    status_label.configure(text=f"Fallback phrase detected: '{recognized_text}'")
                    time.sleep(0.5)
                    if move_to_next_sentence():
                        break
                    else:
                        return

                last_speech_time = time.time()

                for word in recognized_text.split():
                    recognized_words_set.add(clean_text(word))

                if result.is_final:
                    cumulative_text = (cumulative_text + " " + recognized_text).strip() if cumulative_text else recognized_text

                    current_sentence = sentences[current_sentence_index]
                    similarity, match_info = calculate_similarity_detailed(current_sentence, cumulative_text)
                    bert_score_current, current_bert_pass = bert_similarity(cumulative_text, current_sentence, threshold=0.70)

                    update_ui(cumulative_text, similarity, bert_score_current * 100, remain_text)
                    update_word_highlighting(match_info)

                    if check_for_fallback_phrase(cumulative_text):
                        status_label.configure(text=f"Fallback phrase detected in final result: '{cumulative_text}'")
                        time.sleep(0.5)
                        if move_to_next_sentence():
                            break
                        else:
                            return

                    all_words_recognized = all(clean_text(word) in recognized_words_set for word in sentence_words if clean_text(word))

                    match_passed = similarity >= similarity_threshold or current_bert_pass or all_words_recognized

                    if match_passed:
                        status_label.configure(text=f"Sentence {current_sentence_index + 1} completed with classic={similarity:.2f}% / bert={bert_score_current:.2f}")
                        time.sleep(0.5)

                        if current_bert_pass:
                            spoken_words = clean_text(cumulative_text).split()
                            ref_words = clean_text(current_sentence).split()
                            required_ratio = 0.5  

                            if len(spoken_words) < len(ref_words) and len(spoken_words) / len(ref_words) < required_ratio:
                                status_label.configure(text=f"🛑 BERT match too short ({len(spoken_words)}/{len(ref_words)})")
                                continue


                        # Remain text çıkarımı 
                        cleaned_cumulative = clean_text(cumulative_text)
                        cleaned_current = clean_text(current_sentence)

                        if cleaned_current in cleaned_cumulative:
                            remain_text = cleaned_cumulative.replace(cleaned_current, '').strip()
                        else:
                            remain_words = clean_text(cumulative_text).split()
                            for word in clean_text(current_sentence).split():
                                if word in remain_words:
                                    remain_words.remove(word)
                            remain_text = ' '.join(remain_words)

                        update_ui(cumulative_text, similarity, bert_score_current * 100, remain_text)

                        # Bir sonraki cümleyle BERT karşılaştırması
                        if current_sentence_index + 1 < len(sentences):
                            next_sentence = sentences[current_sentence_index + 1]
                            score_next, skip_next = bert_similarity(remain_text, next_sentence)



                                # 🔒 Remain text için minimum kelime kontrolü
                            remain_words = clean_text(remain_text).split()
                            next_words = clean_text(next_sentence).split()
                            required_ratio = 0.5

                            if len(remain_words) < 2 or len(remain_words) / len(next_words) < required_ratio:
                                print(f"⛔ Remain text too short to skip next sentence ({len(remain_words)}/{len(next_words)} words)")
                                skip_next = False

                                
                            # 🧾 Analiz logları
                            print("\n📌 Current sentence:", current_sentence)
                            print("🎙️ Recognized text:", cumulative_text)
                            print(f"📊 Klasik benzerlik: {similarity:.2f}%")
                            print(f"🧠 BERT skoru: {bert_score_current:.2f}")
                            print("✅ Matched → Remain text hesaplanmasına geçiliyor.\n")
                            print("🧩 Remain text:", remain_text)
                            print("🆚 Next sentence:", next_sentence)
                            print(f"🧠 Remain vs Next Sentence → BERT Similarity Score: {score_next:.2f}")
                            if skip_next:
                                print("✅ Remain text semantik olarak eşleşti → Atlama yapılacak.")
                            else:
                                print("⛔ Remain text did NOT match next sentence enough → No skip. Proceeding normally.")

                            if skip_next:
                                current_sentence_index += 1
                                status_label.configure(text=f"[BERT] Skipping sentence {current_sentence_index + 1} due to semantic match")

                        if not move_to_next_sentence():
                            return
                        break
                else:
                    update_ui(recognized_text, 0, 0, remain_text)
                    temp_match_info = {clean_text(word): "exact" if clean_text(word) in recognized_words_set else "none" for word in sentence_words}
                    update_word_highlighting(temp_match_info)

        except Exception as e:
            print(f"Recognition error: {e}")
            time.sleep(1)

    start_button.configure(state=tk.NORMAL)


# Dolgu ifadeleri için çözümü geliştiren yardımcı fonksiyon
def check_for_fallback_phrase(text):
    # Kullanıcının dolgu ifadesi söyleyip söylemediğini kontrol eder
    clean_recognized = clean_text(text.lower())
    
    for phrase in fallback_phrases:
        if phrase.lower() in clean_recognized:
            return True
    return False

def move_to_next_sentence():
    # Bir sonraki cümleye geçer
    global current_sentence_index, recognized_words_set, progress_pause
    
    status_label.configure(text=f"Moving to next sentence...")
    current_sentence_index += 1
    
    if current_sentence_index < len(sentences):
        # Sonraki cümle için günceller
        current_sentence = sentences[current_sentence_index]
        current_sentence_words = current_sentence.split()
        sentence_words.clear()
        sentence_words.extend(current_sentence_words)
        recognized_words_set.clear()
        update_word_display()
        
        # Cümle ilerlemesini kısa süre duraklatır (hatalı tetiklemeleri önlemek için)
        progress_pause = True
        root.after(2000, lambda: setattr(globals()['progress_pause'], 'value', False))
        
        # Sonraki cümle için hazır olduğumuzu bildirir
        speech_ready.set()
        return True
    else:
        status_label.configure(text="All sentences completed!")
        recognition_running = False
        speech_ready.set()
        start_button.configure(state=tk.NORMAL)
        return False

def speak_text():
    global current_sentence_index, recognition_running, tts_speaking, tts_end_time
    speech_ready.set()
    
    while current_sentence_index < len(sentences) and recognition_running:
        speech_ready.wait()
        speech_ready.clear()
        
        if current_sentence_index >= len(sentences) or not recognition_running:
            break
        
        current_sentence = sentences[current_sentence_index]
        status_label.configure(text=f"Speaking sentence {current_sentence_index + 1}/{len(sentences)}...")

        # Deepgram'dan gelen hız değeri doğrudan kullanır
        rate_value = user_rhythm["speaking_rate"]

        # SSML sadece pause içerir
        ssml_text = create_ssml(current_sentence, pause_ms=user_rhythm["pause_ms"])
        synthesis_input = tts.SynthesisInput(ssml=ssml_text)

        voice = tts.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=tts.SsmlVoiceGender.NEUTRAL
        )
        
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            speaking_rate=rate_value
        )

        try:
            tts_speaking = True

            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            wav = np.frombuffer(response.audio_content, dtype=np.int16)

            word_count = len(current_sentence.split())
            estimated_tts_duration = (word_count / (rate_value * 150)) * 60
            tts_end_time = time.time() + estimated_tts_duration

            sd.play(wav, samplerate=24000)
            sd.wait()

            tts_speaking = False
            time.sleep(0.5)

        except Exception as e:
            print(f"TTS error: {e}")
            tts_speaking = False

def update_text():
    global sentences, current_sentence_index, recognized_words_set, sentence_words, recognition_running
    
    # Devam eden tanımayı durdurur
    recognition_running = False
    
    # Yeni metni alır
    new_text = text_input.get(1.0, tk.END).strip()
    
    # Cümlelere ayırır
    sentences = chunk_by_character_limit(new_text, max_chars=40)
    
    # Dolgu ifadelerini günceller
    update_fallback_phrases()
    
    # Her şeyi sıfırlar
    current_sentence_index = 0
    recognized_words_set.clear()
    
    if sentences:
        # İlk cümle için günceller
        current_sentence = sentences[current_sentence_index]
        sentence_words.clear()
        sentence_words.extend(current_sentence.split())
        update_word_display()
        update_ui("", 0)
        status_label.configure(text="Text updated. Ready to start.")
        start_button.configure(state=tk.NORMAL)
    else:
        status_label.configure(text="No valid sentences found in the text.")


def start_recognition():
    global recognition_running, current_sentence_index, silence_timer_active
    global recognize_thread_ref, speak_thread_ref  

    if recognition_running:
        recognition_running = False
        time.sleep(1)

    if current_sentence_index >= len(sentences):
        current_sentence_index = 0
        update_text()

    update_fallback_phrases()
    start_button.configure(state=tk.DISABLED)

    recognition_running = True

    # >>> Silence timer aktif olur sadece burada
    silence_timer_active = True
    update_silence_timer()

    recognize_thread_ref = threading.Thread(target=recognize_speech, daemon=True)
    speak_thread_ref = threading.Thread(target=speak_text, daemon=True)

    recognize_thread_ref.start()
    speak_thread_ref.start()


def stop_program():
    global recognition_running, silence_timer_active

    recognition_running = False
    silence_timer_active = False

    try:
        if stream.is_active():
            stream.stop_stream()
        if stream.is_stopped() == False:
            stream.close()
    except Exception as e:
        print(f"🎤 Stream already closed or failed to stop: {e}")

    try:
        p.terminate()
    except Exception as e:
        print(f"🔌 PyAudio termination error: {e}")

    try:
        audio_queue.queue.clear()
        audio_queue.put(b"\x00" * 320)
    except Exception as e:
        print(f"📤 Queue cleanup error: {e}")

    try:
        root.quit()
        root.destroy()
    except Exception as e:
        print(f"🪟 Tkinter shutdown error: {e}")





# --- Ek Özellikler --- #
def show_chunk_preview(original_text_data):
    # İçeride max_chars'ı sormak için özel bir CTkToplevel kullanır
    ask_window = ctk.CTkToplevel()
    ask_window.title("Chunk Size")
    ask_window.geometry("400x200")
    ask_window.configure(fg_color="#1e1d23")

    label = ctk.CTkLabel(ask_window, text="Enter maximum chunk length (e.g. 40):", font=("Calibri", 16), text_color="#ffffff")
    label.pack(pady=20)

    entry = ctk.CTkEntry(ask_window, placeholder_text="40", font=("Calibri", 16), width=200, fg_color="#2a292f", text_color="#ffffff")
    entry.pack()

    result = {"value": None}

    def on_ok():
        try:
            val = int(entry.get())
            if 20 <= val <= 200:
                result["value"] = val
                ask_window.destroy()
                open_preview_window(val)  # doğrudan chunk ekranını açar
            else:
                label.configure(text="⚠ Please enter a number between 20 and 200.")
        except:
            label.configure(text="⚠ Invalid input. Please enter a number.")

    def on_cancel():
        ask_window.destroy()

    button_frame = ctk.CTkFrame(ask_window, fg_color="transparent")
    button_frame.pack(pady=20)

    ok_btn = ctk.CTkButton(button_frame, text="OK", command=on_ok, fg_color="#6661ea", hover_color="#9592e9", text_color="black", width=100)
    ok_btn.pack(side=tk.LEFT, padx=10)

    cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=on_cancel, fg_color="#6661ea", hover_color="#9592e9", text_color="black", width=100)
    cancel_btn.pack(side=tk.LEFT, padx=10)

    ask_window.grab_set()


def open_preview_window(max_chars):
    # Chunk edit penceresi
    preview_window = ctk.CTkToplevel()
    preview_window.title("Chunk Preview")
    preview_window.geometry("600x500")
    preview_window.configure(fg_color="#1e1d23")

    if isinstance(original_text, list):
        chunks = original_text
    else:
        chunks = chunk_by_character_limit(original_text, max_chars)

    title_label = ctk.CTkLabel(
        preview_window,
        text="You can edit the chunks below:",
        font=("Courier", 16, "bold"),
        text_color="#ffffff"
    )
    title_label.pack(pady=10)

    text_frame = ctk.CTkFrame(preview_window, fg_color="#2a292f", corner_radius=8)
    text_frame.pack(fill="both", expand=True, padx=15, pady=10)

    edit_box = ctk.CTkTextbox(
        text_frame,
        font=("Calibri", 16),
        wrap=tk.WORD,
        fg_color="#2a292f",
        text_color="#ffffff",
        scrollbar_button_color="#444",
        scrollbar_button_hover_color="#666"
    )
    edit_box.pack(fill="both", expand=True, padx=10, pady=10)
    edit_box.insert(tk.END, "\n".join(chunks))

    def save_chunks():
        global changed_chunks
        changed_chunks = [line.strip() for line in edit_box.get("1.0", tk.END).splitlines() if line.strip()]
        print("✅ Chunks updated.")
        preview_window.destroy()

    save_button = ctk.CTkButton(
        preview_window,
        text="Save",
        command=save_chunks,
        fg_color="#6661ea",
        hover_color="#9592e9",
        text_color="black"
    )
    save_button.pack(pady=10)



def run_calibration():
    global user_rhythm, tts_rhythm_label

    popup = tk.Toplevel()
    popup.title("Rhythm Calibration")
    popup.geometry("430x220")

    instruction = tk.Label(popup, text="🎤 Please read the sentence clearly:\n\n'The quick brown fox jumps over the lazy dog'", font=("Calibri", 15), wraplength=400)
    instruction.pack(pady=10)

    status = tk.Label(popup, text="Click the button below to start recording.", font=("Calibri", 13))
    status.pack()

    result_label = tk.Label(popup, text="", font=("Calibri", 14, "bold"))
    result_label.pack(pady=5)

    def start_recording():
        global user_rhythm
        cal_status.configure(text="🔴 Recording... Please speak now.")
        popup.update()

        record_sample_until_silence("calibration.wav")

        status.configure(text="📤 Sending to Deepgram...")
        popup.update()

        result = analyze_with_deepgram("calibration.wav")
        print("Deepgram response received ✅")
        print(result)  # ← Deepgram çıktısını terminalde gösterir

        rhythm = extract_rhythm_from_deepgram(result)
        user_rhythm = rhythm

        # Sonuçları UI'a yazar
        result_label.configure(text=f"✅ Calibrated!\nRate: {rhythm['speaking_rate']:.2f} | Pause: {rhythm['pause_ms']} ms")
        status.configure(text="✔ Calibration completed.")
        tts_rhythm_label.configure(text=f"TTS Rhythm: {rhythm['speaking_rate']:.2f} / {rhythm['pause_ms']}ms")

    record_button = ttk.Button(popup, text="🎙 Start Recording", command=start_recording)
    record_button.pack(pady=10)





# Test butonu
def skip_to_next():
    # Test amaçlı bir sonraki cümleye geçiş butonu
    if recognition_running:
        move_to_next_sentence()




frame_page1 = ctk.CTkFrame(root, fg_color="black")
frame_page2 = ctk.CTkFrame(root, fg_color="black")
frame_page3 = ctk.CTkFrame(root, fg_color="black")

def reset_session_state():
    global recognition_running, last_speech_time, progress_pause, tts_speaking, tts_end_time, sentence_completed, silence_timer_active
    global recognize_thread_ref, speak_thread_ref  

    recognition_running = False
    silence_timer_active = False
    last_speech_time = time.time()
    progress_pause = False
    tts_speaking = False
    tts_end_time = 0
    sentence_completed = False

    # THREADLERİN DURMASINI BEKLER
    if recognize_thread_ref is not None:
        recognize_thread_ref.join(timeout=1)
        recognize_thread_ref = None

    if speak_thread_ref is not None:
        speak_thread_ref.join(timeout=1)
        speak_thread_ref = None

# --- Page navigation function ---
def show_page(index):
    global recognition_running

    frame_page1.pack_forget()
    frame_page2.pack_forget()
    frame_page3.pack_forget()


    # Sayfa geçişi yaparken (herhangi bir sayfaya geçerken), sistem çalışıyorsa durdur
    if recognition_running:
        print("⚡️ Page changed → stopping recognition and TTS threads.")
        recognition_running = False
        reset_session_state()

    # --- -------------------- --- #

    if index == 1:
        frame_page1.pack(fill=tk.BOTH, expand=True)
    elif index == 2:
        frame_page2.pack(fill=tk.BOTH, expand=True)
    elif index == 3:
        global current_sentence_index, current_sentence, sentence_words, recognized_words_set

        current_sentence_index = 0
        recognized_words_set.clear()

        if sentences:
            current_sentence = sentences[current_sentence_index]
            sentence_words.clear()
            sentence_words.extend(current_sentence.split())
            sentence_label.configure(text=f"Current Sentence ({current_sentence_index + 1}/{len(sentences)}): {current_sentence}")

        update_word_display()
        update_fallback_display()
        update_full_text_display()

        frame_page3.pack(fill=tk.BOTH, expand=True)


def next_page_and_update(next_page_index):
    global current_sentence, sentence_words
    
    # First update the text from page 1 if coming from there
    if frame_page1.winfo_ismapped():
        update_text()


    if next_page_index == 2:
        update_chunk_editor()

    if next_page_index == 3:
        global sentences
        if changed_chunks:
            sentences = changed_chunks.copy()
        else:
            sentences = chunk_by_character_limit(original_text, max_chars=40)


    if next_page_index == 3:
        if current_sentence_index < len(sentences):
            current_sentence = sentences[current_sentence_index]
            sentence_words.clear()
            sentence_words.extend(current_sentence.split())
            
            sentence_label.configure(text=f"Current Sentence ({current_sentence_index + 1}/{len(sentences)}): {current_sentence}")

    show_page(next_page_index)

    if next_page_index == 3:
        update_word_display()
        root.update()

# ========================== PAGE 1 ==========================
from tkinter import filedialog

def upload_txt_file():
    global original_text
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            text_input.delete(1.0, tk.END)
            text_input.insert(tk.END, file_content)
            original_text = file_content

def save_text_and_fallbacks():
    global original_text, sentences, changed_chunks  # ← changed_chunks EKLENİR
    original_text = text_input.get(1.0, tk.END).strip()
    sentences = chunk_by_character_limit(original_text, max_chars=40)

    changed_chunks = []  

    update_fallback_phrases()
    status_label_page1.config(text="✅ Text and fallback words saved.")


frame_page1.configure(fg_color="#121117")

# --- Başlık ---
title_label = ctk.CTkLabel(
    frame_page1,
    text="SmartPrompter",
    font=("Courier", 24, "bold"),
    text_color="#ffffff"
)
title_label.pack(pady=10)

# === MAIN TEXT BLOCK ===
main_shadow = ctk.CTkFrame(frame_page1, fg_color="#0f0f13")
main_shadow.pack(pady=(12, 0), padx=50, fill=tk.BOTH, expand=True)

main_input_frame = ctk.CTkFrame(main_shadow, fg_color="#1e1d23", corner_radius=10)
main_input_frame.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

main_input_label = ctk.CTkLabel(
    main_input_frame,
    text="Type the text below",
    font=("Courier", 14, "bold"),
    text_color="#ffffff",
    fg_color="#1e1d23",
    anchor="w"
)
main_input_label.pack(fill=tk.X, padx=10, pady=(10, 0))

text_input = ctk.CTkTextbox(
    main_input_frame,
    font=("Calibri", 16),
    wrap=tk.WORD,
    height=8,
    fg_color="#2a292f",
    text_color="#ffffff",
    scrollbar_button_color="#444",
    scrollbar_button_hover_color="#666",
    corner_radius=8
)
text_input.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
text_input.insert("0.0", text)

# --- TXT Yükleme Alanı ---
upload_frame = tk.Frame(frame_page1, bg="#121117")
upload_frame.pack(pady=10)

upload_label = tk.Label(
    upload_frame,
    text="Or upload a TXT file:",
    font=("Courier", 12, "bold"),
    fg="#d8dee9",
    bg="#121117"
)
upload_label.pack(side=tk.LEFT, padx=5)

upload_button = ctk.CTkButton(
    upload_frame,
    text="Upload TXT",
    command=upload_txt_file,
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black"
)
upload_button.pack(side=tk.LEFT)

# === FALLBACK BLOCK ===
fallback_shadow = ctk.CTkFrame(frame_page1, fg_color="#0f0f13")
fallback_shadow.pack(pady=(12, 0), padx=50, fill=tk.BOTH, expand=True)

fallback_input_frame = ctk.CTkFrame(fallback_shadow, fg_color="#1e1d23", corner_radius=10)
fallback_input_frame.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

fallback_label = ctk.CTkLabel(
    fallback_input_frame,
    text="Type the fallback words below",
    font=("Courier", 14, "bold"),
    text_color="#ffffff",
    fg_color="#1e1d23",
    anchor="w"
)
fallback_label.pack(fill=tk.X, padx=10, pady=(10, 0))

fallback_text = ctk.CTkTextbox(
    fallback_input_frame,
    font=("Calibri", 16),
    wrap=tk.WORD,
    height=8,
    fg_color="#2a292f",
    text_color="#ffffff",
    scrollbar_button_color="#444",
    scrollbar_button_hover_color="#666",
    corner_radius=8
)
fallback_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
fallback_text.insert("0.0", "\n".join(fallback_phrases))

# --- Butonlar ---
button_container = tk.Frame(frame_page1, bg="#121117")
button_container.pack(pady=10)

save_button = ctk.CTkButton(
    button_container,
    text="Save",
    command=save_text_and_fallbacks,
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=180
)
save_button.pack(pady=4)

chunk_button_page1 = ctk.CTkButton(
    button_container,
    text="Change the Chunks",
    command=lambda: show_chunk_preview(original_text),
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=180
)
chunk_button_page1.pack(pady=4)

next_button1 = ctk.CTkButton(
    button_container,
    text="Next",
    command=lambda: next_page_and_update(2),
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=180
)
next_button1.pack(pady=4)

status_label_page1 = tk.Label(
    button_container,
    text="",
    font=("Courier", 10),
    fg="#0CF035",
    bg="#121117"
)
status_label_page1.pack()



# ========================== PAGE 2 ==========================

# Aktif kartı tutmak için KULLANILIR
active_card = None

def set_active_card(card):
    global active_card
    if active_card:
        active_card.configure(border_color="#2a292f", border_width=2)
    active_card = card
    card.configure(border_color="#9592e9", border_width=3)  # Mor kenar

# Kart oluşturma fonksiyonu
def create_card(title):
    # Kartı CTkFrame olarak oluştur, koyu arka plana sahip   
    card = ctk.CTkFrame(cards_inner, fg_color="#1e1d23", corner_radius=8)
    card.configure(border_width=2, border_color="#2a292f")
    card.pack(side=tk.LEFT, padx=15, pady=20)
    card.configure(width=380, height=500)
    card.pack_propagate(False)

    # Kart başlığı: CTkLabel, beyaz/metin rengi #ffffff
    header = ctk.CTkLabel(
        card,
        text=title,
        font=("Calibri", 20, "bold"),
        text_color="#ffffff",
        fg_color="#1e1d23"
    )
    header.pack(pady=(10, 5))

    # Kart içeriğini tutacak alt frame (iç çerçeve):
    inner_frame = ctk.CTkFrame(card, fg_color="#1e1d23", corner_radius=0)
    inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    inner_frame.pack_propagate(False)

    # Kart seçildiğinde kenarlığı yeşile döner
    def on_click(event):
        set_active_card(card)

    card.bind("<Button-1>", on_click)
    header.bind("<Button-1>", on_click)
    inner_frame.bind("<Button-1>", on_click)

    return card, inner_frame

# 1. frame_page2'yi koyu arka planla tanımlar
frame_page2.configure(fg_color="#121117")  

# Kartları taşıyan dış çerçeveyi (container) CTkFrame olarak oluşturur
cards_container = ctk.CTkFrame(frame_page2, fg_color="#121117")
cards_container.pack(fill=tk.BOTH, expand=True)

# Kartları ortalamak için iç çerçeveyi CTkFrame olarak oluşturur
cards_inner = ctk.CTkFrame(cards_container, fg_color="#121117")
cards_inner.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Kartlar
cal_card, cal_inner = create_card("🎙 Rhythm Calibration")
test_card, test_inner = create_card("🎧 Test Rhythm")
chunk_card, chunk_inner = create_card("✂️ Edit Chunks")

visualizer = EmbeddedWaveformVisualizer(cal_inner)


# --- Kart içerikleri ---

# Calibration card
cal_label = ctk.CTkLabel(
    cal_inner,
    text="Read the sentence below:",
    font=("Calibri", 14),
    text_color="#ffffff",
    fg_color="#1e1d23"
)
cal_label.pack(anchor="w", padx=5, pady=(5, 0))

cal_sent = ctk.CTkLabel(
    cal_inner,
    text="'The quick brown fox jumps over the lazy duck'",
    font=("Calibri", 14, "italic"),
    text_color="#ffffff",
    fg_color="#1e1d23"
)
cal_sent.pack(anchor="w", padx=5, pady=(0, 10))

cal_status = ctk.CTkLabel(
    cal_inner,
    text="",
    font=("Calibri", 14),
    text_color="#ffffff",
    fg_color="#1e1d23"
)
cal_status.pack(anchor="w", padx=5)

def run_inline_calibration():
    global user_rhythm

    def calibration_thread():
        global user_rhythm

        visualizer.start()  # Grafik başlar
        cal_status.configure(text="🔴 Recording... Please speak now.")
        root.update()

        record_sample_until_silence("calibration.wav")  # BLOKLAMASIN diye thread içinde

        visualizer.stop()  # Grafik durur

        cal_status.configure(text="📤 Sending to Deepgram...")
        root.update()

        result = analyze_with_deepgram("calibration.wav")
        rhythm = extract_rhythm_from_deepgram(result)
        user_rhythm = rhythm

        cal_status.configure(text=f"✅ Rate={rhythm['speaking_rate']:.2f}, Pause={rhythm['pause_ms']}ms")
        tts_rhythm_label.configure(text=f"TTS Rhythm: {rhythm['speaking_rate']:.2f} / {rhythm['pause_ms']}ms")

    threading.Thread(target=calibration_thread, daemon=True).start()




ctk.CTkButton(
    cal_inner,
    text="Start Recording",
    command=run_inline_calibration,
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=200
).place(relx=0.5, rely=1.0, anchor="s", y=-10)

# Test rhythm card
ctk.CTkLabel(
    test_inner,
    text="This is a test of your current speaking rhythm.",
    font=("Calibri", 14),
    text_color="#ffffff",
    fg_color="#1e1d23"
).pack(anchor="w", padx=5, pady=(5, 10))

ctk.CTkButton(
    test_inner,
    text="Play Sample",
    command=test_rhythm_playback,
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=200
).place(relx=0.5, rely=1.0, anchor="s", y=-10)

# Chunk edit card
chunk_editor = ctk.CTkTextbox(
    chunk_inner,
    font=("Calibri", 16),
    wrap=tk.WORD,
    fg_color="#2a292f",
    text_color="#ffffff",
    scrollbar_button_color="#444",
    scrollbar_button_hover_color="#666",
    corner_radius=8,
    width=280,
    height=300
)
chunk_editor.pack(expand=True, fill=tk.BOTH, padx=5, pady=(5, 0))
chunk_editor.insert(tk.END, "\n".join(sentences))

def apply_chunks():
    global changed_chunks
    changed_chunks = [line.strip() for line in chunk_editor.get(1.0, tk.END).splitlines() if line.strip()]
    print("✅ Chunks updated.")

ctk.CTkButton(
    chunk_inner,
    text="Apply Chunks",
    command=apply_chunks,
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=200
).pack(pady=10)

# Navigation frame (en altta ortalanır)
nav_frame2 = ctk.CTkFrame(frame_page2, fg_color="transparent")
nav_frame2.pack(side=tk.BOTTOM, pady=30)

back_button2 = ctk.CTkButton(
    nav_frame2,
    text="Back",
    command=lambda: show_page(1),
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=150
)
back_button2.pack(side=tk.LEFT, padx=20)

next_button2 = ctk.CTkButton(
    nav_frame2,
    text="Next",
    command=lambda: next_page_and_update(3),
    fg_color="#6661ea",
    hover_color="#9592e9",
    text_color="black",
    width=150
)
next_button2.pack(side=tk.LEFT, padx=20)

# ========================== PAGE 3 ==========================

frame_page3 = ctk.CTkFrame(root, fg_color="#121117", height=600)
frame_page3.pack_propagate(False)

# Sentence display
sentence_frame = ctk.CTkFrame(frame_page3, fg_color="transparent")
sentence_frame.pack(pady=20, fill=tk.X)

sentence_label = ctk.CTkLabel(
    sentence_frame,
    text="Current Sentence:",
    font=("Calibri", 16, "bold"),
    text_color="#ffffff"
)
sentence_label.pack(anchor=tk.W, padx=10)

word_frame = ctk.CTkFrame(frame_page3, fg_color="transparent")
word_frame.pack(pady=10, fill=tk.X)

# Status indicators
status_frame = ctk.CTkFrame(frame_page3, fg_color="transparent")
status_frame.pack(pady=10, fill=tk.X)

status_label = ctk.CTkLabel(status_frame, text="Ready to start...", font=("Calibri", 16),text_color="#ffffff")
status_label.pack(side=tk.LEFT, padx=10)

similarity_label = ctk.CTkLabel(status_frame, text="Similarity: 0%", font=("Calibri", 16),text_color="#ffffff")
similarity_label.pack(side=tk.LEFT, padx=10)

bert_label = ctk.CTkLabel(status_frame, text="BERT: 0%", font=("Calibri", 16),text_color="#ffffff")
bert_label.pack(side=tk.LEFT, padx=10)

silence_label = ctk.CTkLabel(status_frame, text="Silence: 0s", font=("Calibri", 16),text_color="#ffffff")
silence_label.pack(side=tk.LEFT, padx=10)

tts_rhythm_label = ctk.CTkLabel(status_frame, text=f"TTS Rhythm: {user_rhythm['speaking_rate']:.2f} / {user_rhythm['pause_ms']}ms", font=("Calibri", 16),text_color="#ffffff")
tts_rhythm_label.pack(side=tk.LEFT, padx=10)

# Recognized Speech
recognized_frame = ctk.CTkFrame(frame_page3, fg_color="#1e1d23", corner_radius=8)
recognized_frame.pack(pady=10, padx=10, fill=tk.X, expand=False)

recognized_title = ctk.CTkLabel(
    recognized_frame,
    text="Recognized Speech",
    font=("Calibri", 16, "bold"),
    text_color="#ffffff"
)
recognized_title.pack(anchor="w", padx=10, pady=(5, 0))

recognized_text_area = ctk.CTkTextbox(
    recognized_frame,
    font=("Calibri", 16),
    wrap=tk.WORD,
    height=90,
    fg_color="#2a292f",
    text_color="#ffffff",
    scrollbar_button_color="#444",
    scrollbar_button_hover_color="#666",
    corner_radius=8
)
recognized_text_area.pack(fill=tk.X, expand=False, padx=10, pady=5)

"""
# Remain Text
remain_frame = ctk.CTkFrame(frame_page3, fg_color="#1e1d23", corner_radius=8)
remain_frame.pack(pady=10, padx=10, fill=tk.X)

remain_title = ctk.CTkLabel(
    remain_frame,
    text="Remain Text for Next Sentence (BERT check)",
    font=("Calibri", 16, "bold"),
    text_color="#ffffff"
)
remain_title.pack(anchor="w", padx=10, pady=(5, 0))

remain_text_area = scrolledtext.ScrolledText(
    remain_frame,
    font=("Calibri", 16),
    wrap=tk.WORD,
    height=3
)
remain_text_area.configure(bg="#2a292f", fg="#ffffff", insertbackground="#ffffff")
remain_text_area.pack(fill=tk.X, padx=10, pady=5)
"""

# Color Legend
legend_frame = ctk.CTkFrame(frame_page3, fg_color="#1e1d23", corner_radius=8)
legend_frame.pack(pady=10, padx=10, fill=tk.X)

legend_title = ctk.CTkLabel(
    legend_frame,
    text="Color Legend",
    font=("Calibri", 16, "bold"),
    text_color="#ffffff"
)
legend_title.pack(anchor="w", padx=10, pady=(5, 0))

legend_items = [
    ("Exact Match", "green"),
    ("Homophone Match", "orange"),
    ("Semantic Match", "blue"),
    ("No Match", "white"),
    ("Numeric Match", "green")
]

for text, color in legend_items:
    item_frame = ctk.CTkFrame(legend_frame, fg_color="transparent")
    item_frame.pack(side=tk.LEFT, padx=10, pady=5)

    color_label = tk.Label(item_frame, width=2, height=1, bg=color)
    color_label.pack(side=tk.LEFT)

    desc_label = ctk.CTkLabel(
        item_frame,
        text=text,
        font=("Calibri", 15),
        text_color="#ffffff"
    )
    desc_label.pack(side=tk.LEFT, padx=5)

# === Dolgu Kelimeleri ve Tüm Metin Alanı ===

bottom_frame = ctk.CTkFrame(frame_page3, fg_color="transparent")
bottom_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=False)

# --- Fallback Frame ---
fallback_frame = ctk.CTkFrame(bottom_frame, fg_color="#1e1d23", corner_radius=8)
fallback_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

fallback_title = ctk.CTkLabel(
    fallback_frame,
    text="Fallback Phrases",
    font=("Calibri", 16, "bold"),
    text_color="#ffffff"
)
fallback_title.pack(anchor="w", padx=10, pady=(5, 0))

fallback_display = ctk.CTkTextbox(
    fallback_frame,
    font=("Calibri", 16),
    wrap=tk.WORD,
    height=180,
    fg_color="#2a292f",
    text_color="#ffffff",
    scrollbar_button_color="#444",
    scrollbar_button_hover_color="#666",
    corner_radius=8
)
fallback_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)


# --- Full Text Frame ---
fulltext_frame = ctk.CTkFrame(bottom_frame, fg_color="#1e1d23", corner_radius=8)
fulltext_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

fulltext_title = ctk.CTkLabel(
    fulltext_frame,
    text="Full Text",
    font=("Calibri", 16, "bold"),
    text_color="#ffffff"
)
fulltext_title.pack(anchor="w", padx=10, pady=(5, 0))

full_text_display = ctk.CTkTextbox(
    fulltext_frame,
    font=("Calibri", 16),
    wrap=tk.WORD,
    height=180,
    fg_color="#2a292f",
    text_color="#ffffff",
    scrollbar_button_color="#444",
    scrollbar_button_hover_color="#666",
    corner_radius=8
)
full_text_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Controls
button_container3 = ctk.CTkFrame(frame_page3, fg_color="transparent")
button_container3.pack(pady=20)

button_frame3 = ctk.CTkFrame(button_container3, fg_color="transparent")
button_frame3.pack()

# 3 butonlu 1 satırlık grid
start_button = ctk.CTkButton(
    button_frame3, text="Start", command=start_recognition,
    fg_color="#6661ea", hover_color="#9592e9", text_color="black", width=160
)
start_button.grid(row=0, column=0, padx=10)

stop_button = ctk.CTkButton(
    button_frame3, text="Stop", command=stop_program,
    fg_color="#6661ea", hover_color="#9592e9", text_color="black", width=160
)
stop_button.grid(row=0, column=1, padx=10)

skip_button = ctk.CTkButton(
    button_frame3, text="Skip to Next", command=skip_to_next,
    fg_color="#6661ea", hover_color="#9592e9", text_color="black", width=160
)
skip_button.grid(row=0, column=2, padx=10)

# Back button
back_button3 = ctk.CTkButton(
    frame_page3,
    text="Back",
    command=lambda: [reset_session_state(), show_page(2)],
    fg_color="#6661ea", hover_color="#9592e9", text_color="black", width=160
)
back_button3.pack(pady=5)


# ==== Güncelleme fonksiyonları (Sayfa 3'e geçerken çağırılır) ====

def update_fallback_display():
    fallback_display.configure(state="normal")
    fallback_display.delete(1.0, tk.END)
    for phrase in fallback_phrases:
        fallback_display.insert(tk.END, phrase + "\n")
    fallback_display.configure(state="disabled")

def update_full_text_display():
    full_text_display.configure(state="normal")
    full_text_display.delete(1.0, tk.END)

    if changed_chunks:
        full_text_display.insert(tk.END, "\n".join(changed_chunks))
    else:
        full_text_display.insert(tk.END, original_text)
    
    full_text_display.configure(state="disabled")


# Ana döngüyü başlatır
show_page(1)
root.protocol("WM_DELETE_WINDOW", stop_program)  # Kapatırken temizler
root.mainloop()