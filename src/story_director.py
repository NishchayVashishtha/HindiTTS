"""
story_director.py
-----------------
AI Story Narration Studio:
Two-Agent Director that transforms raw Hindi stories into full audio dramas:
  - Agent 01 (Script Analyst): Segments story into Narrator vs Character dialogue turns.
  - Agent 02 (Acoustic Director): Assigns voice profiles, pacing, and emotions.
Then synthesizes all segments and joins them into a complete multi-voice audio story.
"""

import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from openai import OpenAI
from src.tts_engine import HindiTTSEngine

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

AGENT_01_SYSTEM = """
You are Agent 01 — Script Analyst for a Hindi Audio Story System.
Split the provided Hindi story into distinct dialogue and narration turns.
Return ONLY a valid JSON array of objects:
[
  {
    "turn_id": 1,
    "type": "narrator" | "character",
    "speaker": "Narrator" | "<Character Name in English>",
    "gender": "male" | "female",
    "text": "<Exact Hindi text for this turn>"
  }
]
"""

AGENT_02_SYSTEM = """
You are Agent 02 — Acoustic Director for Hindi TTS.
For each turn in the JSON list, assign optimal acoustic delivery attributes:
- "emotion": "neutral" | "happy" | "scared" | "angry" | "mysterious" | "gentle"
- "voice_profile": "narrator_male" | "narrator_female" | "pintu_bhaiya" | "elder_dadi_maa" | "horror_ghost" | "hero_energetic"
- "rate": "-20%" to "+20%"
- "pitch": "-20Hz" to "+20Hz"

Return ONLY the updated JSON array with these additional acoustic fields.
"""

class StoryDirector:
    def __init__(self, output_dir: str | Path = None):
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY is not set in .env")
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "openai/gpt-oss-120b"
        self.tts = HindiTTSEngine(output_dir=output_dir or (BASE_DIR / "outputs"))

    def analyze_story(self, story_text: str) -> list[dict]:
        """Runs Agent 01 and Agent 02 to create an acoustic screenplay."""
        # 1. Agent 01: Turn Splitting
        resp_1 = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_01_SYSTEM},
                {"role": "user", "content": f"Hindi Story:\n{story_text}"}
            ],
            temperature=0.1
        )
        content_1 = resp_1.choices[0].message.content
        match_1 = re.search(r'\[.*\]', content_1, re.DOTALL)
        if not match_1:
            raise ValueError(f"Agent 01 JSON parsing failed: {content_1[:200]}")
        turns = json.loads(match_1.group(0))

        # 2. Agent 02: Acoustic Direction
        resp_2 = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_02_SYSTEM},
                {"role": "user", "content": f"Turns JSON:\n{json.dumps(turns, ensure_ascii=False)}"}
            ],
            temperature=0.2
        )
        content_2 = resp_2.choices[0].message.content
        match_2 = re.search(r'\[.*\]', content_2, re.DOTALL)
        if match_2:
            try:
                turns = json.loads(match_2.group(0))
            except Exception:
                pass
        return turns

    def narrate_story(self, story_text: str, project_id: str = "story_demo") -> dict:
        """Analyzes story and synthesizes audio for every turn."""
        turns = self.analyze_story(story_text)
        generated_clips = []

        for i, turn in enumerate(turns):
            clip_name = f"{project_id}_turn_{i+1:02d}.mp3"
            voice_prof = turn.get("voice_profile", "narrator_male")
            
            # Map gender if profile not set
            if voice_prof == "narrator_male" and turn.get("gender") == "female":
                voice_prof = "narrator_female"

            clip_path = self.tts.generate_character_speech(
                text=turn["text"],
                character_profile=voice_prof,
                filename=clip_name
            )
            turn["audio_file"] = str(clip_path)
            generated_clips.append(clip_path)

        return {
            "turns": turns,
            "clips": generated_clips
        }

if __name__ == "__main__":
    director = StoryDirector()
    sample = "एक घने जंगल में एक छोटा भालू रहता था। उसने कहा, 'अरे वाह, कितना मीठा शहद है!' फिर वह खुशी से नाचने लगा।"
    res = director.narrate_story(sample)
    print(f"Generated {len(res['clips'])} narration turns.")
