#!/usr/bin/env python3
"""
10_clean_dataset.py
-------------------
Phase 2 — Data Quality Filter

Reads  : data/metadata.csv  (output of 04_transcribe_whisper.py)
Writes : data/metadata_clean.csv  (high-quality segments only)

Filters applied:
  1. Duration  : 2.0 s – 15.0 s  (ideal for TTS training)
  2. Text      : Non-empty, >= 5 Hindi characters, no gibberish
  3. SNR       : Signal-to-Noise Ratio >= 15 dB
  4. Duplicates: Remove exact-text duplicates (keep longest audio)
  5. Lang check: Must contain >= 1 Devanagari Unicode character

Usage:
    python scripts/10_clean_dataset.py
    python scripts/10_clean_dataset.py --snr-threshold 20 --min-dur 3

Dependencies:
    pip install pandas numpy soundfile tqdm
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
INPUT_CSV     = BASE_DIR / "data" / "metadata.csv"
OUTPUT_CSV    = BASE_DIR / "data" / "metadata_clean.csv"
REJECTED_CSV  = BASE_DIR / "data" / "metadata_rejected.csv"

# ── Devanagari Unicode range ──────────────────────────────────────────────────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# ── Gibberish / noise indicators in Whisper output ───────────────────────────
NOISE_PATTERNS = re.compile(
    r'\[.*?\]|<.*?>|\(.*?\)|'           # [Music] <noise> (inaudible)
    r'\b(hmm|uh|um|ah|oh|eh)\b',        # filler words
    re.IGNORECASE
)


def compute_snr(audio: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio using the top 90th percentile as
    'signal' and bottom 10th percentile as 'noise'.
    Returns dB value; higher = cleaner audio.
    """
    if len(audio) == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return 0.0
    # Use RMS of loudest 10% vs quietest 10% as proxy
    abs_audio = np.abs(audio)
    signal_rms = np.sqrt(np.mean(abs_audio[abs_audio >= np.percentile(abs_audio, 90)] ** 2))
    noise_rms  = np.sqrt(np.mean(abs_audio[abs_audio <= np.percentile(abs_audio, 10)] ** 2))
    if noise_rms == 0:
        return 60.0   # effectively infinite SNR — silent noise floor
    snr_db = 20 * np.log10(signal_rms / noise_rms)
    return round(snr_db, 2)


def is_valid_hindi_text(text: str, min_chars: int = 5) -> bool:
    """Check text has sufficient Devanagari characters and no noise markers."""
    text = text.strip()
    if not text or len(text) < min_chars:
        return False
    devanagari_chars = len(DEVANAGARI_RE.findall(text))
    if devanagari_chars < 2:
        return False
    cleaned = NOISE_PATTERNS.sub('', text).strip()
    if len(cleaned) < 3:
        return False
    return True


def _resolve_path(raw: str) -> Path | None:
    """Resolve audio path — handles both absolute and cross-machine paths."""
    p = Path(raw)
    if p.exists():
        return p
    # Try relative to data dir
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "data":
            candidate = BASE_DIR / Path(*parts[i:])
            if candidate.exists():
                return candidate
    return None


def clean_dataset(
    snr_threshold: float = 15.0,
    min_duration: float = 2.0,
    max_duration: float = 15.0,
    min_text_chars: int = 5,
) -> None:

    if not INPUT_CSV.exists():
        print(f"[-] metadata.csv not found: {INPUT_CSV}")
        print("    Run scripts 01–04 first.")
        return

    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"\n{'='*60}")
    print(f"🧹 Dataset Cleaning — {total} input rows")
    print(f"{'='*60}")
    print(f"   Thresholds: SNR >= {snr_threshold} dB | "
          f"duration {min_duration}–{max_duration} s | "
          f"text >= {min_text_chars} chars")
    print()

    # ── Normalise column names ─────────────────────────────────────────────────
    df.columns = [c.strip().lower() for c in df.columns]
    text_col = next((c for c in df.columns if 'text' in c), None)
    path_col = next((c for c in df.columns if 'audio' in c or 'path' in c), None)
    if not text_col or not path_col:
        print(f"[-] Required columns not found. Got: {list(df.columns)}")
        return

    rejected_rows = []
    kept_rows     = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        reason = None
        text   = str(row.get(text_col, "")).strip()
        raw_path = str(row.get(path_col, ""))

        # ── 1. Text quality ────────────────────────────────────────────────────
        if not is_valid_hindi_text(text, min_text_chars):
            reason = "text_invalid"

        # ── 2. Audio file exists ───────────────────────────────────────────────
        audio_path = None
        if reason is None:
            audio_path = _resolve_path(raw_path)
            if audio_path is None:
                reason = "file_missing"

        # ── 3. Duration + SNR ─────────────────────────────────────────────────
        snr = None
        duration = None
        if reason is None:
            try:
                info     = sf.info(str(audio_path))
                duration = info.frames / info.samplerate
                if duration < min_duration or duration > max_duration:
                    reason = f"duration_{duration:.1f}s"
                else:
                    audio, _ = sf.read(str(audio_path))
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    snr = compute_snr(audio.astype(np.float32))
                    if snr < snr_threshold:
                        reason = f"low_snr_{snr:.1f}dB"
            except Exception as e:
                reason = f"read_error"

        if reason:
            row_dict = row.to_dict()
            row_dict["reject_reason"] = reason
            rejected_rows.append(row_dict)
        else:
            row_dict = row.to_dict()
            row_dict["audio_path"] = str(audio_path)
            row_dict["duration"]   = round(duration, 3)
            row_dict["snr_db"]     = snr
            kept_rows.append(row_dict)

    # ── 4. Remove duplicate texts (keep longest audio) ─────────────────────────
    before_dedup = len(kept_rows)
    kept_df = pd.DataFrame(kept_rows)
    if not kept_df.empty and text_col in kept_df.columns:
        kept_df = (
            kept_df
            .sort_values("duration", ascending=False)
            .drop_duplicates(subset=[text_col], keep="first")
            .reset_index(drop=True)
        )
    after_dedup = len(kept_df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(
            REJECTED_CSV, index=False, encoding="utf-8-sig"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    kept    = len(kept_df)
    removed = total - kept

    print(f"\n{'='*60}")
    print(f"✅ Cleaning Complete!")
    print(f"{'='*60}")
    print(f"   Input rows         : {total}")
    print(f"   Kept (clean)       : {kept}  ({kept/total*100:.1f}%)")
    print(f"   Removed            : {removed}  ({removed/total*100:.1f}%)")
    print(f"     - Duplicates     : {before_dedup - after_dedup}")
    if kept_df.empty:
        print("   ⚠️  No clean samples found — check your thresholds!")
    else:
        dur_col = kept_df.get("duration", pd.Series(dtype=float))
        print(f"   Avg duration       : {dur_col.mean():.2f} s")
        print(f"   Total duration     : {dur_col.sum()/60:.1f} min")
        if "snr_db" in kept_df.columns:
            print(f"   Avg SNR            : {kept_df['snr_db'].mean():.1f} dB")
    print(f"\n   Clean CSV  → {OUTPUT_CSV}")
    if rejected_rows:
        print(f"   Rejected CSV → {REJECTED_CSV}")
    print(f"{'='*60}\n")

    # ── Rejection breakdown ───────────────────────────────────────────────────
    if rejected_rows:
        rej_df = pd.DataFrame(rejected_rows)
        print("Rejection breakdown:")
        for reason, count in rej_df["reject_reason"].value_counts().items():
            print(f"  {reason:30s}: {count}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Clean Hindi TTS dataset")
    parser.add_argument("--snr-threshold", type=float, default=15.0,
                        help="Minimum SNR in dB (default: 15)")
    parser.add_argument("--min-dur", type=float, default=2.0,
                        help="Minimum segment duration in seconds (default: 2)")
    parser.add_argument("--max-dur", type=float, default=15.0,
                        help="Maximum segment duration in seconds (default: 15)")
    parser.add_argument("--min-chars", type=int, default=5,
                        help="Minimum Devanagari characters in text (default: 5)")
    args = parser.parse_args()

    clean_dataset(
        snr_threshold=args.snr_threshold,
        min_duration=args.min_dur,
        max_duration=args.max_dur,
        min_text_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
