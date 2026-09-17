import streamlit as st
import os
import tempfile
from moviepy import VideoFileClip
from groq import Groq
from google import genai
from google.genai import types
import srt
from datetime import timedelta
from pydub import AudioSegment

st.set_page_config(page_title="Video Translator AI", layout="wide", page_icon="🎬")

st.title("🎬 Video Speech Translator App")
st.caption("Unggah video, transkrip percakapan, dan terjemahkan secara otomatis!")

# Sidebar Config
st.sidebar.header("🔑 API Configurations")

default_groq = st.secrets.get("GROQ_API_KEY", "")
default_gemini = st.secrets.get("GEMINI_API_KEY", "")

groq_api_key = st.sidebar.text_input("Groq API Key", value=default_groq, type="password", help="Dapatkan gratis di console.groq.com")
gemini_api_key = st.sidebar.text_input("Gemini API Key", value=default_gemini, type="password", help="Dapatkan gratis di aistudio.google.com")

target_language = st.sidebar.selectbox(
    "Pilih Bahasa Target Terjemahan:",
    ["Indonesian", "English", "Japanese", "Spanish", "French", "German", "Korean", "Mandarin"]
)

uploaded_file = st.file_uploader("Pilih file video (.mp4, .mov, .avi, .mkv)", type=["mp4", "mov", "avi", "mkv"])

def transcribe_audio_file(client_groq, audio_file_path):
    """Fungsi helper untuk mentranskripsi satu file audio via Groq Whisper"""
    with open(audio_file_path, "rb") as audio_file:
        transcription = client_groq.audio.transcriptions.create(
            file=(audio_file_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
    return transcription.segments

def process_video_translation(video_path):
    client_groq = Groq(api_key=groq_api_key)
    client_gemini = genai.Client(api_key=gemini_api_key)

    # 1. Ekstraksi Audio dari Video
    st.info("🎵 1/3: Memisahkan & mengompres audio dari video...")
    audio_path = video_path.replace(os.path.splitext(video_path)[1], ".mp3")
    
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        audio_path, 
        bitrate="64k", 
        logger=None
    )
    clip.close()

    # 2. Transkripsi dengan Groq Whisper (Auto-Chunking jika > 20 MB)
    st.info("🎙️ 2/3: Mentranskripsi suara (Speech-to-Text)...")
    
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    all_segments = []

    if file_size_mb > 20:
        st.warning(f"Ukuran audio ({file_size_mb:.1f} MB) melebihi batas 20 MB. Memotong audio menjadi beberapa bagian...")
        
        # Load audio dengan pydub
        sound = AudioSegment.from_file(audio_path)
        # Potong per 10 menit (600,000 ms)
        chunk_length_ms = 10 * 60 * 1000
        chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
        
        time_offset = 0.0  # Waktu offset dalam detik untuk penyesuaian timestamp
        
        for idx, chunk in enumerate(chunks):
            st.text(f"--- Memproses bagian audio {idx + 1} dari {len(chunks)} ---")
            chunk_path = f"{audio_path}_chunk_{idx}.mp3"
            chunk.export(chunk_path, format="mp3", bitrate="64k")
            
            # Transkripsi chunk
            segments = transcribe_audio_file(client_groq, chunk_path)
            
            # Sesuaikan timestamp dengan offset
            for seg in segments:
                seg.start += time_offset
                seg.end += time_offset
                all_segments.append(seg)
            
            # Update offset untuk chunk berikutnya
            time_offset += (len(chunk) / 1000.0)
            
            # Hapus chunk temporary
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    else:
        all_segments = transcribe_audio_file(client_groq, audio_path)

    # 3. Penerjemahan dengan Gemini API
    st.info(f"🌐 3/3: Menerjemahkan ke bahasa {target_language}...")
    
    full_text_to_translate = "\n".join([f"[{i}] {seg.text.strip()}" for i, seg in enumerate(all_segments)])
    
    prompt = f"""Kamu adalah penerjemah profesional. Terjemahkan kalimat-kalimat berikut ke dalam bahasa {target_language}.
Jaga format penomoran [x] di awal setiap baris agar sesuai dengan kalimat aslinya! Jangan ubah nomor atau menambah penjelasan lain.

Kalimat:
{full_text_to_translate}
"""
    
    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    # Parse hasil terjemahan
    translated_lines = response.text.strip().split("\n")
    translated_dict = {}
    for line in translated_lines:
        if line.startswith("[") and "]" in line:
            try:
                idx_str, text = line.split("]", 1)
                idx = int(idx_str.replace("[", "").strip())
                clean_text = text.strip().lstrip(". ").strip()
                translated_dict[idx] = clean_text
            except:
                continue

    # 4. Buat File SRT Subtitle
    srt_subtitles = []
    for i, seg in enumerate(all_segments):
        start_time = timedelta(seconds=seg.start)
        end_time = timedelta(seconds=seg.end)
        text = translated_dict.get(i, seg.text.strip())
        
        srt_subtitles.append(
            srt.Subtitle(index=i+1, start=start_time, end=end_time, content=text)
        )
    
    srt_output = srt.compose(srt_subtitles)

    # Clean up audio temp utama
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return srt_output, all_segments, translated_dict

if uploaded_file is not None:
    # Simpan sementara video yang diupload
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    st.video(uploaded_file)

    if st.button("🚀 Mulai Terjemahkan Video", type="primary"):
        if not groq_api_key or not gemini_api_key:
            st.error("Silakan masukkan Groq API Key dan Gemini API Key terlebih dahulu!")
        else:
            with st.spinner("Sedang memproses... Harap tunggu sebentar."):
                try:
                    srt_content, original_segments, translated_dict = process_video_translation(temp_video_path)
                    st.success("✅ Proses Terjemahan Selesai!")

                    # Kolom Download & Preview
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📥 Download Subtitle (.srt)")
                        st.download_button(
                            label="Download File .SRT",
                            data=srt_content,
                            file_name=f"translated_{target_language}.srt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        st.subheader("📜 Hasil Transkrip & Terjemahan")
                        for i, seg in enumerate(original_segments):
                            orig = seg.text.strip()
                            trans = translated_dict.get(i, "-")
                            st.markdown(f"**[{seg.start:.1f}s - {seg.end:.1f}s]**")
                            st.markdown(f"- 🗣️ *Original:* {orig}")
                            st.markdown(f"- 🌐 *Terjemahan:* {trans}")
                            st.divider()

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
            
            # Hapus file temporary video
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
