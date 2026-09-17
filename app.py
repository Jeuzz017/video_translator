import streamlit as st
import os
import tempfile
from moviepy import VideoFileClip
from groq import Groq
from google import genai
from google.genai import types
import srt
from datetime import timedelta

st.set_page_config(page_title="Video Translator AI", layout="wide", page_icon="🎬")

st.title("🎬 Video Speech Translator App")
st.caption("Unggah video, transkrip percakapan, dan terjemahkan secara otomatis!")

# Sidebar Config
st.sidebar.header("🔑 API Configurations")

# Membaca dari Streamlit Secrets jika ada, jika tidak ada baru membaca dari sidebar
default_groq = st.secrets.get("GROQ_API_KEY", "")
default_gemini = st.secrets.get("GEMINI_API_KEY", "")

groq_api_key = st.sidebar.text_input("Groq API Key", value=default_groq, type="password", help="Dapatkan gratis di console.groq.com")
gemini_api_key = st.sidebar.text_input("Gemini API Key", value=default_gemini, type="password", help="Dapatkan gratis di aistudio.google.com")

target_language = st.sidebar.selectbox(
    "Pilih Bahasa Target Terjemahan:",
    ["Indonesian", "English", "Japanese", "Spanish", "French", "German", "Korean", "Mandarin"]
)

uploaded_file = st.file_uploader("Pilih file video (.mp4, .mov, .avi, .mkv)", type=["mp4", "mov", "avi", "mkv"])

def process_video_translation(video_path):
    client_groq = Groq(api_key=groq_api_key)
    client_gemini = genai.Client(api_key=gemini_api_key)

    # 1. Ekstraksi Audio
    st.info("🎵 1/3: Memisahkan audio dari video...")
    audio_path = video_path.replace(os.path.splitext(video_path)[1], ".mp3")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    # 2. Transkripsi dengan Groq Whisper
    st.info("🎙️ 2/3: Mentranskripsi suara (Speech-to-Text)...")
    with open(audio_path, "rb") as audio_file:
        transcription = client_groq.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
    
    segments = transcription.segments

    # 3. Penerjemahan dengan Gemini API
    st.info(f"🌐 3/3: Menerjemahkan ke bahasa {target_language}...")
    
    # PERBAIKAN: Menggunakan seg.text (bukan seg['text'])
    full_text_to_translate = "\n".join([f"[{i}] {seg.text.strip()}" for i, seg in enumerate(segments)])
    
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
                # Clean up titik atau spasi di awal kalimat terjemahan jika ada
                clean_text = text.strip().lstrip(". ").strip()
                translated_dict[idx] = clean_text
            except:
                continue

    # 4. Buat File SRT Subtitle
    srt_subtitles = []
    for i, seg in enumerate(segments):
        # PERBAIKAN: Menggunakan seg.start, seg.end, dan seg.text
        start_time = timedelta(seconds=seg.start)
        end_time = timedelta(seconds=seg.end)
        text = translated_dict.get(i, seg.text.strip())
        
        srt_subtitles.append(
            srt.Subtitle(index=i+1, start=start_time, end=end_time, content=text)
        )
    
    srt_output = srt.compose(srt_subtitles)

    # Clean up audio temp
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return srt_output, segments, translated_dict

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
                            # PERBAIKAN: Menggunakan seg.text, seg.start, seg.end
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
