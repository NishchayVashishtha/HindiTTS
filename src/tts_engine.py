"""
tts_engine.py
-------------
Core Hindi Text-to-Speech (TTS) Engine.
Supports high-fidelity Neural Hindi voices (Male / Female),
dynamic pitch/rate modulations for character voices, and audio export.
"""

import asyncio
from pathlib import Path
import edge_tts

# Pre-configured voice profiles for character / story narration
VOICE_PROFILES = {
    "narrator_male": {
        "voice": "hi-IN-MadhurNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "label": "Narrator (Male - Madhur)",
    },
    "narrator_female": {
        "voice": "hi-IN-SwaraNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "label": "Narrator (Female - Swara)",
    },
    "pintu_bhaiya": {
        "voice": "hi-IN-MadhurNeural",
        "rate": "+15%",
        "pitch": "+15Hz",
        "label": "Pintu Bhaiya (Energetic / Comedic)",
    },
    "elder_dadi_maa": {
        "voice": "hi-IN-SwaraNeural",
        "rate": "-15%",
        "pitch": "-10Hz",
        "label": "Dadi Maa / Elder (Gentle / Slow)",
    },
    "horror_ghost": {
        "voice": "hi-IN-MadhurNeural",
        "rate": "-25%",
        "pitch": "-25Hz",
        "label": "Horror / Demon (Deep / Slow)",
    },
    "hero_energetic": {
        "voice": "hi-IN-MadhurNeural",
        "rate": "+10%",
        "pitch": "+8Hz",
        "label": "Hero / Protagonist (Confident)",
    },
}

class HindiTTSEngine:
    def __init__(self, output_dir: str | Path = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _synthesize_async(
        self,
        text: str,
        output_file: Path,
        voice: str = "hi-IN-MadhurNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        """Asynchronously synthesizes speech using Microsoft Neural Edge-TTS."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")
            
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        await communicate.save(str(output_file))
        return output_file

    def generate_speech(
        self,
        text: str,
        filename: str = "output.mp3",
        voice: str = "hi-IN-MadhurNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        """Synchronous wrapper to generate speech."""
        output_path = self.output_dir / filename
        asyncio.run(
            self._synthesize_async(
                text=text,
                output_file=output_path,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
        )
        return output_path

    def generate_character_speech(
        self,
        text: str,
        character_profile: str = "narrator_male",
        filename: str = "character_speech.mp3",
    ) -> Path:
        """Generates speech using a pre-tuned acoustic profile."""
        profile = VOICE_PROFILES.get(character_profile, VOICE_PROFILES["narrator_male"])
        return self.generate_speech(
            text=text,
            filename=filename,
            voice=profile["voice"],
            rate=profile["rate"],
            pitch=profile["pitch"],
        )

if __name__ == "__main__":
    engine = HindiTTSEngine()
    out = engine.generate_speech(
        "नमस्ते, आपका स्वागत है हमारे हिंदी टेक्स्ट-टू-स्पीच प्रोजेक्ट में।",
        filename="welcome.mp3",
    )
    print(f"Generated test audio at: {out}")

