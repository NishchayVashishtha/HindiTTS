import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Hindi TTS & Story Narration Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.tts_engine import HindiTTSEngine, VOICE_PROFILES
from src.story_director import StoryDirector

# Initialize engine
@st.cache_resource
def get_tts_engine():
    return HindiTTSEngine(output_dir=BASE_DIR / "outputs")

tts_engine = get_tts_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Hindi%20TTS-Production%20Ready-brightgreen?style=for-the-badge&logo=python", use_container_width=True)
    st.title("🎙️ Hindi TTS Studio")
    st.markdown("""
    **End-to-End Hindi Text-to-Speech & AI Narration Pipeline**
    - ⚡ **Neural Engine:** Edge-TTS & MMS
    - 🎭 **AI Director:** Multi-speaker Story Dramatization
    - 📁 **Dataset:** 26 Hindi Story Audio Collections
    - 🚀 **Hardware:** Hybrid Local CPU / Cloud GPU (Colab)
    """)
    st.divider()
    st.caption("Developed for Hindi Audio Storytelling & TTS Model Research")

# Main Header
st.title("🎙️ Hindi Text-to-Speech (TTS) & AI Story Narration Studio")
st.markdown("Generate ultra-realistic Hindi speech, dramatize multi-character stories, and inspect the Google Drive training dataset.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗣️ Instant Hindi TTS",
    "🎭 Multi-Speaker Story Narrator",
    "📁 Google Drive Dataset Hub (26 Stories)",
    "🚀 Cloud & Multi-Device Training Guide"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Instant Hindi TTS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Single Speaker Hindi Speech Synthesis")
    col1, col2 = st.columns([2, 1])

    with col1:
        default_text = "नमस्ते! आपका स्वागत है हमारे हिंदी टेक्स्ट-टू-स्पीच स्टूडियो में। आप किसी भी वाक्य को यहाँ आवाज़ में बदल सकते हैं।"
        user_text = st.text_area(
            "हिंदी टेक्स्ट दर्ज करें (Enter Hindi Text):",
            value=default_text,
            height=140
        )

    with col2:
        voice_choice = st.selectbox(
            "आवाज़ का प्रकार (Voice Profile):",
            options=list(VOICE_PROFILES.keys()),
            format_func=lambda x: VOICE_PROFILES[x]["label"]
        )
        rate_val = st.slider("बोलने की गति (Speech Rate %):", min_value=-50, max_value=50, value=0, step=5)
        pitch_val = st.slider("पिच (Pitch Hz):", min_value=-30, max_value=30, value=0, step=2)

    if st.button("🔊 आवाज़ उत्पन्न करें (Generate Speech)", type="primary"):
        if not user_text.strip():
            st.warning("कृपया कोई टेक्स्ट लिखें।")
        else:
            with st.spinner("आवाज़ बनाई जा रही है..."):
                profile = VOICE_PROFILES[voice_choice]
                rate_str = f"{rate_val:+d}%" if rate_val != 0 else profile["rate"]
                pitch_str = f"{pitch_val:+d}Hz" if pitch_val != 0 else profile["pitch"]
                
                output_path = tts_engine.generate_speech(
                    text=user_text,
                    filename="instant_tts.mp3",
                    voice=profile["voice"],
                    rate=rate_str,
                    pitch=pitch_str
                )
                
                st.success("✅ ऑडियो सफलतापूर्वक तैयार हो गया!")
                st.audio(str(output_path), format="audio/mp3")
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ डाउनलोड ऑडियो (Download MP3)",
                        data=f,
                        file_name="hindi_speech.mp3",
                        mime="audio/mp3"
                    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Multi-Speaker Story Narrator
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("AI Story Narration Studio (Autonomous Screenplay & Voice Acting)")
    st.markdown("AI आपके लिखे नाटक या कहानी को **पात्रों (Characters) और सूत्रधार (Narrator)** में विभाजित करके हर पात्र को अलग आवाज़ और भावना (Emotion) देगा।")

    story_default = """एक घने जंगल में एक छोटा शेर का बच्चा रहता था। अचानक शाम को जोर से बिजली कड़की और तेज बारिश होने लगी।
शेर का बच्चा कांपने लगा और उसने अपनी मां से डरते हुए कहा, "मां, मुझे बहुत डर लग रहा है!"
मां शेरनी ने उसे सहलाते हुए प्यार से कहा, "डरो मत मेरे बच्चे, जब तक मैं तुम्हारे साथ हूं, कोई तुम्हें नुकसान नहीं पहुंचा सकता।"
शेर के बच्चे ने मुस्कुरा कर कहा, "सच में मां? तुम सबसे बहादुर हो!"
और फिर दोनों शांति से गुफा में सो गए।"""

    input_story = st.text_area("अपनी कहानी यहाँ लिखें (Enter Hindi Story):", value=story_default, height=180)

    if st.button("🎬 कहानी का ऑडियो ड्रामा बनाएं (Direct & Narrate Story)", type="primary"):
        with st.spinner("AI निर्देशक कहानी का विश्लेषण कर रहा है और आवाज़ें डब कर रहा है..."):
            try:
                director = StoryDirector(output_dir=BASE_DIR / "outputs")
                result = director.narrate_story(input_story, project_id="web_drama")
                
                st.success(f"✅ कहानी को {len(result['turns'])} दृश्यों में डब कर दिया गया है!")
                
                st.markdown("### 📜 AI Screenplay & Character Audio")
                for i, turn in enumerate(result["turns"], 1):
                    with st.expander(f"दृश्य {i}: [{turn.get('speaker', 'Unknown')}] — {turn.get('emotion', 'neutral').upper()} भाव", expanded=True):
                        st.markdown(f"**संवाद (Dialogue):** {turn['text']}")
                        st.markdown(f"*आवाज़ प्रोफ़ाइल:* `{turn.get('voice_profile', 'default')}` | *लिंग:* `{turn.get('gender', 'male')}`")
                        if "audio_file" in turn and Path(turn["audio_file"]).exists():
                            st.audio(turn["audio_file"], format="audio/mp3")

            except Exception as e:
                st.error(f"कहानी डब करने में त्रुटि: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Google Drive Dataset Hub
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📁 Google Drive Dataset Explorer (26 Audio Stories)")
    st.markdown("""
    **Shared Folder:** [hindi_tts_data (Google Drive)](https://drive.google.com/drive/folders/17OKMSInQ7-8KFNUn0voAYPhFzViLAFra?usp=sharing)  
    यह डेटासेट 26 हाई-क्वालिटी हिंदी कहानियों का संकलन है जिसका उपयोग TTS मॉडल्स को फाइन-ट्यून करने के लिए किया जाता है।
    """)

    raw_dir = BASE_DIR / "data" / "01_raw_audio"
    wav_files = list(raw_dir.glob("*.wav")) if raw_dir.exists() else []

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("कुल कहानियाँ (Total Stories in Drive)", "26 Files")
    col_stat2.metric("स्थानीय रूप से डाउनलोड (Locally Synced)", f"{len(wav_files)} / 26")
    total_size_mb = sum(f.stat().st_size for f in wav_files) / (1024 * 1024) if wav_files else 0
    col_stat3.metric("कुल साइज़ (Total Size)", f"{total_size_mb:.1f} MB")

    if wav_files:
        st.markdown("### 🎧 डाउनलोड की गई कहानियाँ (Downloaded Audio Stories)")
        selected_file = st.selectbox("सुनने के लिए कहानी चुनें (Select Audio to Play):", options=wav_files, format_func=lambda x: x.name)
        if selected_file:
            st.audio(str(selected_file), format="audio/wav")
            file_mb = selected_file.stat().st_size / (1024 * 1024)
            st.caption(f"फ़ाइल का नाम: `{selected_file.name}` | साइज़: `{file_mb:.2f} MB`")
    else:
        st.info("फ़ाइलें अभी बैकग्राउंड में डाउनलोड हो रही हैं। डाउनलोड पूरा होते ही यहाँ दिखाई देंगी।")

    # Diarized segments metadata preview
    meta_path = BASE_DIR / "data" / "metadata.csv"
    if meta_path.exists():
        st.markdown("### 📋 Whisper ट्रांसक्रिप्शन एवं डाइराइजेशन मेटाडेटा (Metadata Preview)")
        df_meta = pd.read_csv(meta_path)
        st.dataframe(df_meta[["speaker_id", "text", "story"]].head(15), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Cloud & Multi-Device Training Hub
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🚀 Maximum Compute Training Guide (Google Colab T4/A100 GPU)")
    st.markdown("""
    > [!IMPORTANT]
    > **Hardware Notice:** Local Windows मशीन पर सिर्फ़ CPU उपलब्ध है। 26 बड़ी ऑडियो फ़ाइल्स पर डीप लर्निंग TTS मॉडल (GPT-SoVITS / XTTS) को ट्रेन करने के लिए **Google Colab Free T4 GPU** सबसे तेज़ और आसान तरीका है।
    """)

    st.markdown("""
    ### 3 आसान स्टेप्स में Google Colab पर ट्रेन करें:
    
    1. **Google Colab खोलें:** [Google Colab](https://colab.research.google.com/) पर जाएं और नया नोटबुक बनाएं।
    2. **Runtime GPU चुनें:** `Runtime` -> `Change runtime type` -> `T4 GPU` सेलेक्ट करें।
    3. **नीचे दिया गया कोड कॉपी-पेस्ट करें और Run करें:**
    """)

    colab_code = f"""# ==============================================================================
# Google Colab 1-Click Hindi TTS Training Pipeline
# Hardware: Free NVIDIA T4 GPU
# Dataset: hindi_tts_data from Google Drive
# ==============================================================================

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Check GPU
import torch
print("GPU Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 3. Clone Repository & Install Dependencies
!git clone https://github.com/paarth-error/hindi_tts.git /content/HindiTTS
%cd /content/HindiTTS
!pip install -q edge-tts yt-dlp openai-whisper soundfile gdown torchaudio

# 4. Copy Audio Files from Google Drive
import os, shutil
drive_folder = '/content/drive/MyDrive/hindi_tts_data'
local_folder = '/content/HindiTTS/data/01_raw_audio'
os.makedirs(local_folder, exist_ok=True)

if os.path.exists(drive_folder):
    for f in os.listdir(drive_folder):
        if f.endswith('.wav'):
            shutil.copy(os.path.join(drive_folder, f), local_folder)
    print("Copied files:", len(os.listdir(local_folder)))
else:
    print("Downloading via gdown...")
    import gdown
    gdown.download_folder("https://drive.google.com/drive/folders/17OKMSInQ7-8KFNUn0voAYPhFzViLAFra?usp=sharing", output=local_folder)

# 5. Start Training / Formatting
!python scripts/08_format_sovits.py --stats
"""
    st.code(colab_code, language="python")

