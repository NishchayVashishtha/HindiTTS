#!/usr/bin/env python3
"""
download_drive_data.py
----------------------
Downloads all Hindi audio story WAV files from the shared Google Drive folder:
https://drive.google.com/drive/folders/17OKMSInQ7-8KFNUn0voAYPhFzViLAFra

Target directory: data/01_raw_audio/
"""

import os
import sys
from pathlib import Path
import gdown

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/17OKMSInQ7-8KFNUn0voAYPhFzViLAFra?usp=sharing"
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "01_raw_audio"

def download_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading Hindi TTS dataset from Google Drive...")
    print(f"[*] Destination: {OUTPUT_DIR}")
    
    try:
        downloaded = gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            output=str(OUTPUT_DIR),
            quiet=False,
            use_cookies=False
        )
        print(f"\n[+] Download completed successfully!")
        print(f"[+] Total files downloaded: {len(downloaded) if downloaded else 'Check directory'}")
        
        # Verify files
        wav_files = list(OUTPUT_DIR.glob("*.wav"))
        print(f"[+] Total .wav files in {OUTPUT_DIR}: {len(wav_files)}")
        for i, f in enumerate(wav_files, 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {i}. {f.name} ({size_mb:.2f} MB)")
            
    except Exception as e:
        print(f"[-] Download error: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    download_dataset()

