#!/usr/bin/env python3
"""
Step 8: Format dataset for different TTS training frameworks.
Supports: GPT-SoVITS, Coqui XTTS, Piper

FIXED: Now reads from BOTH possible manifest locations:
  - data/07_final/manifest.json  (old pipeline output)
  - data/dataset_manifest.json   (06_export_manifest.py output)
  - data/metadata.csv            (direct Whisper output — Colab fallback)

FIXED: XTTS config auto-detects GPU device (cuda / cpu / mps).
FIXED: Audio paths are re-resolved relative to the CURRENT repo root,
       so hardcoded Mac paths in metadata.csv don't break Colab runs.
"""

import os
import json
import argparse
import csv
import soundfile as sf
from pathlib import Path
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# Priority order: look for the manifest in multiple locations
MANIFEST_CANDIDATES = [
    BASE_DIR / "data" / "07_final" / "manifest.json",
    BASE_DIR / "data" / "dataset_manifest.json",
]
METADATA_CSV = BASE_DIR / "data" / "metadata.csv"
SEGMENTS_DIR = BASE_DIR / "data" / "03_diarized_segments"


def detect_device() -> str:
    """Auto-detect best available device for training."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_audio_path(raw_path: str) -> Path | None:
    """
    Resolve an audio path that may be an absolute Mac/Linux path from a
    different machine. Strategy:
      1. Try the path as-is.
      2. Try stripping everything before 'data/' and joining with BASE_DIR.
    """
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    # Strip machine-specific prefix, keep only the data/... tail
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "data":
            relative = Path(*parts[i:])
            candidate = BASE_DIR / relative
            if candidate.exists():
                return candidate.resolve()
    return None


def _load_manifest() -> tuple[list[dict], Path]:
    """
    Load dataset manifest from the first available source.
    Returns (samples_list, output_dir).
    """
    # 1. Try JSON manifests
    for manifest_path in MANIFEST_CANDIDATES:
        if manifest_path.exists():
            print(f"[+] Loading manifest: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Normalise: both {"samples": [...]} and flat [...] are accepted
            samples = data["samples"] if isinstance(data, dict) else data
            output_dir = manifest_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            return samples, output_dir

    # 2. Fallback: build from metadata.csv (direct Whisper output)
    if METADATA_CSV.exists():
        print(f"[!] No JSON manifest found. Falling back to: {METADATA_CSV}")
        samples = []
        with open(METADATA_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_path = row.get("audio_path", "")
                resolved = _resolve_audio_path(raw_path)
                if not resolved:
                    continue
                try:
                    info = sf.info(str(resolved))
                    duration = info.frames / info.samplerate
                except Exception:
                    duration = 0.0
                samples.append({
                    "audio_path": str(resolved),
                    "speaker":    row.get("speaker_id", "SPEAKER_00"),
                    "story":      row.get("story", ""),
                    "language":   row.get("language", "hi"),
                    "text":       row.get("text", "").strip(),
                    "duration":   duration,
                    "emotion":    "Neutral",
                })
        output_dir = BASE_DIR / "data" / "07_final"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save a normalised JSON manifest for future runs
        manifest_out = output_dir / "manifest.json"
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
        print(f"[+] Built manifest with {len(samples)} samples from metadata.csv")
        print(f"[+] Saved normalised manifest → {manifest_out}")
        return samples, output_dir

    return [], BASE_DIR / "data" / "07_final"


# ─────────────────────────────────────────────────────────────────────────────

def format_gpt_sovits(samples: list[dict], output_dir: Path) -> Path:
    """Format for GPT-SoVITS training (pipe-separated filelist)."""
    output_file = output_dir / "sovits_filelist.txt"
    lines = []
    for item in samples:
        audio_path = _resolve_audio_path(item.get("audio_path", ""))
        speaker = item.get("speaker", "SPEAKER_00")
        text = item.get("text", "").strip()
        if not audio_path or not text:
            continue
        lines.append(f"{audio_path}|{speaker}|HI|{text}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ GPT-SoVITS format created: {output_file}")
    print(f"   Total samples: {len(lines)}")
    return output_file


def format_coqui_xtts(samples: list[dict], output_dir: Path) -> Path:
    """Format for Coqui XTTS-v2 training."""
    metadata_file = output_dir / "metadata.csv"
    lines = ["audio_file|text|speaker_name"]
    for item in samples:
        audio_path = _resolve_audio_path(item.get("audio_path", ""))
        speaker = item.get("speaker", "SPEAKER_00")
        text = item.get("text", "").strip()
        if not audio_path or not text:
            continue
        lines.append(f"{audio_path}|{text}|{speaker}")

    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    device = detect_device()
    config = {
        "run_name": "hindi_tts_xtts",
        "model": "xtts",
        "audio": {"sample_rate": 22050, "max_wav_value": 32768.0},
        "datasets": [{
            "name": "hindi_tts",
            "path": str(output_dir.parent),
            "meta_file": "metadata.csv",
            "language": "hi"
        }],
        "training": {
            "batch_size": 4,
            "epochs": 100,
            "learning_rate": 0.0001,
            "mixed_precision": device == "cuda",
            "device": device,
        }
    }
    config_file = output_dir / "xtts_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Coqui XTTS format created:")
    print(f"   Metadata : {metadata_file}")
    print(f"   Config   : {config_file}  (device={device})")
    print(f"   Samples  : {len(lines) - 1}")
    return metadata_file


def format_piper(samples: list[dict], output_dir: Path) -> Path:
    """Format for Piper TTS training."""
    metadata_file = output_dir / "piper_metadata.csv"
    lines = []
    for item in samples:
        audio_path = _resolve_audio_path(item.get("audio_path", ""))
        text = item.get("text", "").strip()
        if not audio_path or not text:
            continue
        try:
            relative_path = audio_path.relative_to(BASE_DIR)
        except ValueError:
            relative_path = audio_path
        lines.append(f"{relative_path}|{text}")

    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n✅ Piper TTS format created: {metadata_file}")
    print(f"   Total samples: {len(lines)}")
    return metadata_file


def show_stats(samples: list[dict]):
    """Print dataset statistics."""
    if not samples:
        print("No samples found!")
        return
    durations = [s.get("duration", 0) for s in samples]
    total_dur = sum(durations)
    speakers = set(s.get("speaker", "unknown") for s in samples)

    print("\n" + "="*60)
    print("📊 Dataset Statistics")
    print("="*60)
    print(f"Total samples  : {len(samples)}")
    print(f"Total duration : {total_dur/60:.1f} min  ({total_dur:.0f} s)")
    print(f"Avg duration   : {total_dur/len(samples):.2f} s")
    print(f"Speakers       : {len(speakers)}  → {', '.join(sorted(speakers))}")
    short  = sum(1 for d in durations if d < 3)
    medium = sum(1 for d in durations if 3 <= d < 10)
    long   = sum(1 for d in durations if d >= 10)
    print(f"\nDuration buckets:")
    print(f"  < 3 s  : {short}")
    print(f"  3–10 s : {medium}")
    print(f"  ≥10 s  : {long}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Format Hindi TTS dataset for training")
    parser.add_argument("--format", choices=["sovits", "xtts", "piper", "all"],
                        default="sovits", help="Output format (default: sovits)")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    print("="*60)
    print("🎤 Format Dataset for TTS Training")
    print(f"   Repo root : {BASE_DIR}")
    print("="*60)

    samples, output_dir = _load_manifest()
    if not samples:
        print("\n❌ No dataset found. Run scripts 01–06 first (or check paths).")
        return

    if args.stats:
        show_stats(samples)
        return

    if args.format in ("sovits", "all"):
        format_gpt_sovits(samples, output_dir)
    if args.format in ("xtts", "all"):
        format_coqui_xtts(samples, output_dir)
    if args.format in ("piper", "all"):
        format_piper(samples, output_dir)

    print("\n💡 Next steps:")
    if args.format in ("sovits", "all"):
        print("   GPT-SoVITS: python scripts/09_train_gpt_sovits_mac.py")
    if args.format in ("xtts", "all"):
        print("   XTTS: tts-train --config_path data/07_final/xtts_config.json")
    if args.format in ("piper", "all"):
        print("   Piper: See TRAINING_GUIDE.md for instructions")
    print("\n📖 Read TRAINING_GUIDE.md for detailed instructions")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()