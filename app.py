import streamlit as st
import os
import tempfile
import subprocess
import json
import time
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
    ["English", "Indonesian", "Japanese", "Spanish", "French", "German", "Korean", "Mandarin"]
)

uploaded_file = st.file_uploader("Pilih file video (.mp4, .mov, .avi, .mkv)", type=["mp4", "mov", "avi", "mkv"])

def get_video_duration(video_path):
    """Mendapatkan durasi video (dalam detik) menggunakan ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def extract_audio_chunk(video_path, output_audio_path, start_sec, duration_sec):
    """Memotong & mengekstrak audio langsung dengan FFmpeg CLI"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration_sec),
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "64k",
        "-ac", "1",
        "-ar", "16000",
        output_audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def transcribe_audio_file(client_groq, audio_file_path):
    """Mentranskripsi file audio via Groq Whisper API"""
    with open(audio_file_path, "rb") as audio_file:
        transcription = client_groq.audio.transcriptions.create(
            file=(audio_file_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
    
    if hasattr(transcription, "segments"):
        raw_segments = transcription.segments
    elif isinstance(transcription, dict):
        raw_segments = transcription.get("segments", [])
    else:
        raw_segments = getattr(transcription, "segments", [])

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

def translate_batch_json(client_gemini, batch_items, target_lang):
    """Meminta Gemini mengembalikan hasil terjemahan dalam format JSON murni"""
    prompt = f"""You are a professional translator. 
Translate the 'text' field of each item in the JSON array below into {target_lang}.

Input JSON:
{json.dumps(batch_items, ensure_ascii=False)}

IMPORTANT:
- Return ONLY a valid JSON array containing objects with keys "id" and "translated_text".
- Keep the exact same "id" for each corresponding item.
- Do not add Markdown codeblock formatting like ```json or any explanations.
"""

    models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash']
    
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client_gemini.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                if response and response.text:
                    parsed_json = json.loads(response.text)
                    return parsed_json
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    time.sleep(2 * (attempt + 1))
                    continue
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    break
                else:
                    time.sleep(1)
                    continue
    return []

def process_video_translation(video_path):
    client_groq = Groq(api_key=groq_api_key)
    client_gemini = genai.Client(api_key=gemini_api_key)

    st.info("🎵 1/3: Memisahkan & memeriksa durasi video...")
    
    duration_seconds = get_video_duration(video_path)
    chunk_duration = 600  # 10 menit per bagian
    all_segments = []

    st.info("🎙️ 2/3: Mentranskripsi suara (Speech-to-Text)...")

    if duration_seconds > chunk_duration:
        st.warning(f"Video berdurasi panjang ({duration_seconds/60:.1f} menit). Memproses audio dalam beberapa bagian...")
        
        start_time = 0.0
        chunk_idx = 1
        
        while start_time < duration_seconds:
            current_duration = min(chunk_duration, duration_seconds - start_time)
            st.text(f"--- Memproses bagian {chunk_idx} ({start_time/60:.1f} m - {(start_time+current_duration)/60:.1f} m) ---")
            
            chunk_audio_path = video_path.replace(os.path.splitext(video_path)[1], f"_chunk_{chunk_idx}.mp3")
            
            extract_audio_chunk(video_path, chunk_audio_path, start_time, current_duration)
            
            segments = transcribe_audio_file(client_groq, chunk_audio_path)
            
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
        extract_audio_chunk(video_path, audio_path, 0, duration_seconds)
        all_segments = transcribe_audio_file(client_groq, audio_path)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # 3. Penerjemahan dengan Gemini API (Structured JSON Batching)
    st.info(f"🌐 3/3: Menerjemahkan ke bahasa {target_language}...")
    
    translated_dict = {}
    batch_size = 40
