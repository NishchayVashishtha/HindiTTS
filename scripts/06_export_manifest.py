

import json
import os

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
ENRICHED_CSV     = os.path.join(BASE_DIR, "..", "data", "metadata_enriched.csv")
OUTPUT_MANIFEST  = os.path.join(BASE_DIR, "..", "data", "dataset_manifest.json")


def create_manifest() -> None:
    if not os.path.exists(ENRICHED_CSV):
        print(f"[-] metadata_enriched.csv nahi mili: {os.path.abspath(ENRICHED_CSV)}")
        print("    Pehle 05_llm_autotagging.py run karo.")
        return

    df = pd.read_csv(ENRICHED_CSV)
    print(f"[*] {len(df)} rows found in metadata_enriched.csv")

    manifest = []
    for _, row in df.iterrows():
        entry = {
            "audio_path":   row.get("audio_path", ""),
            "speaker_id":   row.get("speaker_id", ""),
            "story":        row.get("story", ""),
            "language":     row.get("language", "hi"),
            "text":         row.get("text", ""),
            "emotion":      row.get("emotion", "Neutral"),
            "pitch":        row.get("pitch", "Medium"),
            "speed":        row.get("speed", "Normal"),
            "style_prompt": row.get("style_prompt", ""),
        }
        manifest.append(entry)

    # Wrap in {"samples": [...]} so 08_format_sovits.py can consume it directly
    manifest_doc = {"samples": manifest}
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, ensure_ascii=False, indent=2)

    print(f"[+] Dataset manifest ready! Total samples: {len(manifest)}")
    print(f"[+] Saved: {os.path.abspath(OUTPUT_MANIFEST)}")

    # ── Quick stats ───────────────────────────────────────────────────────────
    emotions = [e["emotion"] for e in manifest]
    print("\n── Emotion Distribution ──")
    for emotion, count in pd.Series(emotions).value_counts().items():
        print(f"  {emotion:12s}: {count}")


if __name__ == "__main__":
    create_manifest()
