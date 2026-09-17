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
    # Jika response berupa objek pydantic/dataclass, ubah ke dict agar konsisten
    if hasattr(transcription, "segments"):
        raw_segments = transcription.segments
    elif isinstance(transcription, dict):
        raw_segments = transcription.get("segments", [])
    else:
        raw_segments = getattr(transcription, "segments", [])

    # Konversi setiap segmen ke dictionary murni
    segments_dict_list = []
    for s in raw_segments:
        if isinstance(s, dict):
            segments_dict_list.append(s)
        else:
            segments_dict_list.append({
                "start": getattr(s, "start", 0.0),
                "end": getattr(s, "end", 0.0),
                "text": getattr(s, "text", "")
            })
            
    return segments_dict_list

def process_video_translation(video_path):
    client_groq = Groq(api_key=groq_api_key)
    client_gemini = genai.Client(api_key=gemini_api_key)

    st.info("🎵 1/3: Memisahkan & memeriksa durasi video...")
    
    video_clip = VideoFileClip(video_path)
    duration_seconds = video_clip.duration
    
    chunk_duration = 600  # 10 menit per bagian
    all_segments = []

    st.info("🎙️ 2/3: Mentranskripsi suara (Speech-to-Text)...")

    if duration_seconds > chunk_duration:
        st.warning(f"Video berdurasi panjang ({duration_seconds/60:.1f} menit). Memproses audio dalam beberapa bagian...")
        
        start_time = 0
        chunk_idx = 1
        
        while start_time < duration_seconds:
            end_time = min(start_time + chunk_duration, duration_seconds)
            st.text(f"--- Memproses bagian {chunk_idx} ({start_time/60:.1f} m - {end_time/60:.1f} m) ---")
            
            chunk_audio_path = video_path.replace(os.path.splitext(video_path)[1], f"_chunk_{chunk_idx}.mp3")
            
            sub_clip = video_clip.subclipped(start_time, end_time)
            sub_clip.audio.write_audiofile(chunk_audio_path, bitrate="64k", logger=None)
            sub_clip.close()
            
            segments = transcribe_audio_file(client_groq, chunk_audio_path)
            
            # Sesuaikan timestamp dengan offset durasi
            for seg in segments:
                all_segments.append({
                    "start": seg["start"] + start_time,
                    "end": seg["end"] + start_time,
                    "text": seg["text"]
                })
                
            if os.path.exists(chunk_audio_path):
                os.remove(chunk_audio_path)
                
            start_time += chunk_duration
            chunk_idx += 1
    else:
        audio_path = video_path.replace(os.path.splitext(video_path)[1], ".mp3")
        video_clip.audio.write_audiofile(audio_path, bitrate="64k", logger=None)
        all_segments = transcribe_audio_file(client_groq, audio_path)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)

    video_clip.close()

    # 3. Penerjemahan dengan Gemini API
    st.info(f"🌐 3/3: Menerjemahkan ke bahasa {target_language}...")
    
    full_text_to_translate = "\n".join([f"[{i}] {seg['text'].strip()}" for i, seg in enumerate(all_segments)])
    
    prompt = f"""Kamu adalah penerjemah profesional. Terjemahkan kalimat-kalimat berikut ke dalam bahasa {target_language}.
Jaga format penomoran [x] di awal setiap baris agar sesuai dengan kalimat aslinya! Jangan ubah nomor atau menambah penjelasan lain.

Kalimat:
{full_text_to_translate}
"""
    
    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
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
        start_td = timedelta(seconds=seg["start"])
        end_td = timedelta(seconds=seg["end"])
        text = translated_dict.get(i, seg["text"].strip())
        
        srt_subtitles.append(
            srt.Subtitle(index=i+1, start=start_td, end=end_td, content=text)
        )
    
    srt_output = srt.compose(srt_subtitles)

    return srt_output, all_segments, translated_dict

if uploaded_file is not None:
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
                            orig = seg["text"].strip()
                            trans = translated_dict.get(i, "-")
                            st.markdown(f"**[{seg['start']:.1f}s - {seg['end']:.1f}s]**")
                            st.markdown(f"- 🗣️ *Original:* {orig}")
                            st.markdown(f"- 🌐 *Terjemahan:* {trans}")
                            st.divider()

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
            
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
