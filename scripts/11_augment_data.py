#!/usr/bin/env python3
"""
11_augment_data.py
------------------
Phase 2 — Data Augmentation

Reads  : data/metadata_clean.csv  (output of 10_clean_dataset.py)
Writes : data/metadata_augmented.csv
         data/05_augmented/  (augmented WAV files)

Augmentations applied:
  1. Speed perturbation  : 0.9x, 1.1x  (keeps pitch, changes duration)
  2. Pitch shifting      : ±2 semitones  (keeps speed)
  3. Volume normalization: target -20 dBFS

Usage:
    python scripts/11_augment_data.py
    python scripts/11_augment_data.py --no-pitch --speed-factors 0.85 1.0 1.15

Dependencies:
    pip install soundfile numpy pandas tqdm librosa
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[!] librosa not installed — pitch shifting disabled.")
    print("    Install with: pip install librosa")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
INPUT_CSV     = BASE_DIR / "data" / "metadata_clean.csv"
OUTPUT_CSV    = BASE_DIR / "data" / "metadata_augmented.csv"
AUGMENTED_DIR = BASE_DIR / "data" / "05_augmented"

AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SPEED_FACTORS  = [0.9, 1.1]      # 1.0 = original (always included)
DEFAULT_PITCH_SEMITONES = [-2, 2]         # semitones up/down
TARGET_DBFS            = -20.0            # loudness normalization target


def normalize_volume(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize audio to target dBFS level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    current_dbfs = 20 * np.log10(rms)
    gain = 10 ** ((target_dbfs - current_dbfs) / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def speed_perturb(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """
    Speed perturbation via resampling trick:
    resample to factor*sr then read back at sr → changes speed, keeps pitch.
    """
    if HAS_LIBROSA:
        return librosa.effects.time_stretch(audio, rate=factor)
    # Fallback: simple linear interpolation (lower quality)
    orig_len   = len(audio)
    target_len = int(orig_len / factor)
    indices    = np.linspace(0, orig_len - 1, target_len)
    return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by n semitones without changing duration."""
    if not HAS_LIBROSA:
        return audio   # skip if librosa unavailable
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def augment_dataset(
    speed_factors: list[float],
    pitch_semitones: list[float],
    normalize: bool = True,
) -> None:

    if not INPUT_CSV.exists():
        print(f"[-] {INPUT_CSV} not found.")
        print("    Run scripts/10_clean_dataset.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    total = len(df)

    print(f"\n{'='*60}")
    print(f"🔧 Data Augmentation — {total} clean samples")
    print(f"{'='*60}")
    print(f"   Speed factors   : {speed_factors}")
    print(f"   Pitch semitones : {pitch_semitones}")
    print(f"   Normalize volume: {normalize} (target {TARGET_DBFS} dBFS)")
    multiplier = 1 + len(speed_factors) + (len(pitch_semitones) if HAS_LIBROSA else 0)
    print(f"   Expected output : ~{total * multiplier} samples")
    print()

    records = []

    # ── Copy originals (with optional normalization) ──────────────────────────
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying originals"):
        src = Path(str(row.get("audio_path", "")))
        if not src.exists():
            continue

        try:
            audio, sr = sf.read(str(src))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            if normalize:
                audio = normalize_volume(audio, TARGET_DBFS)

            dest = AUGMENTED_DIR / f"orig_{src.stem}.wav"
            sf.write(str(dest), audio, sr)

            rec = row.to_dict()
            rec["audio_path"] = str(dest)
            rec["augmentation"] = "original"
            records.append(rec)
        except Exception as e:
            print(f"  [-] Skip {src.name}: {e}")

    # ── Speed perturbation ────────────────────────────────────────────────────
    for factor in speed_factors:
        label = f"speed_{factor:.2f}x".replace(".", "p")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Speed {factor}x"):
            src = Path(str(row.get("audio_path", "")))
            if not src.exists():
                continue
            try:
                audio, sr = sf.read(str(src))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                aug   = speed_perturb(audio, sr, factor)
                if normalize:
                    aug = normalize_volume(aug, TARGET_DBFS)
                dest  = AUGMENTED_DIR / f"{label}_{src.stem}.wav"
                sf.write(str(dest), aug, sr)
                rec = row.to_dict()
                rec["audio_path"]  = str(dest)
                rec["augmentation"] = label
                records.append(rec)
            except Exception as e:
                print(f"  [-] Skip {src.name}: {e}")

    # ── Pitch shifting ────────────────────────────────────────────────────────
    if HAS_LIBROSA:
        for semitones in pitch_semitones:
            sign  = "p" if semitones >= 0 else "m"
            label = f"pitch_{sign}{abs(int(semitones))}st"
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Pitch {semitones:+}st"):
                src = Path(str(row.get("audio_path", "")))
                if not src.exists():
                    continue
                try:
                    audio, sr = sf.read(str(src))
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio = audio.astype(np.float32)
                    aug   = pitch_shift(audio, sr, semitones)
                    if normalize:
                        aug = normalize_volume(aug, TARGET_DBFS)
                    dest  = AUGMENTED_DIR / f"{label}_{src.stem}.wav"
                    sf.write(str(dest), aug, sr)
                    rec = row.to_dict()
                    rec["audio_path"]   = str(dest)
                    rec["augmentation"] = label
                    records.append(rec)
                except Exception as e:
                    print(f"  [-] Skip {src.name}: {e}")

    # ── Save augmented metadata ───────────────────────────────────────────────
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"✅ Augmentation complete!")
    print(f"{'='*60}")
    print(f"   Original samples  : {total}")
    print(f"   Augmented total   : {len(records)}")
    print(f"   Multiplier        : {len(records)/total:.1f}x")
    print(f"   Output dir        : {AUGMENTED_DIR}")
    print(f"   Output CSV        : {OUTPUT_CSV}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Augment Hindi TTS dataset")
    parser.add_argument("--speed-factors", nargs="+", type=float,
                        default=DEFAULT_SPEED_FACTORS,
                        help="Speed multipliers (default: 0.9 1.1)")
    parser.add_argument("--pitch-semitones", nargs="+", type=float,
                        default=DEFAULT_PITCH_SEMITONES,
                        help="Pitch shifts in semitones (default: -2 2)")
    parser.add_argument("--no-pitch", action="store_true",
                        help="Disable pitch shifting")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip volume normalization")
    args = parser.parse_args()

    augment_dataset(
        speed_factors=args.speed_factors,
        pitch_semitones=[] if args.no_pitch else args.pitch_semitones,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
