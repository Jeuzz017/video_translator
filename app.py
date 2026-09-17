def process_video_translation(video_path):
    client_groq = Groq(api_key=groq_api_key)
    client_gemini = genai.Client(api_key=gemini_api_key)

    # 1. Ekstraksi & Kompresi Audio
    st.info("🎵 1/3: Memisahkan & mengompres audio dari video...")
    audio_path = video_path.replace(os.path.splitext(video_path)[1], ".mp3")
    
    clip = VideoFileClip(video_path)
    # Menggunakan bitrate 64k agar ukuran file sangat kecil (<25MB)
    clip.audio.write_audiofile(
        audio_path, 
        bitrate="64k", 
        logger=None
    )
    clip.close()

    # 2. Transkripsi dengan Groq Whisper
    st.info("🎙️ 2/3: Mentranskripsi suara (Speech-to-Text)...")
    
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > 24:
        st.warning(f"Ukuran audio ({file_size_mb:.1f} MB) mendekati batas maksimum Groq (25 MB).")

    with open(audio_path, "rb") as audio_file:
        transcription = client_groq.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
    
    segments = transcription.segments

    # 3. Penerjemahan dengan Gemini API
    st.info(f"🌐 3/3: Menerjemahkan ke bahasa {target_language}...")
    
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
                clean_text = text.strip().lstrip(". ").strip()
                translated_dict[idx] = clean_text
            except:
                continue

    # 4. Buat File SRT Subtitle
    srt_subtitles = []
    for i, seg in enumerate(segments):
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
