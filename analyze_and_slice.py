"""
Phase 1: Analyze a folder of audio files and save a library index to disk.
The index is what generate.py loads to build loops from your music.

SETUP (run once):
    pip install librosa soundfile numpy

USAGE:
    python analyze_and_slice.py /path/to/your/music/folder

OUTPUT:
    library_index.pkl  -- your analyzed library, ready for generate.py
"""

import os
import sys
import argparse
import pickle
import librosa
import numpy as np

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a")
HOP_LENGTH = 512   # ~11.6ms per frame at 44.1kHz
N_MFCC = 13        # timbre fingerprint resolution
INDEX_FILE = "library_index.pkl"


def find_audio_files(folder):
    paths = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(AUDIO_EXTENSIONS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def analyze_track(path):
    """Analyze one track and return a list of slice records -- one per beat.

    Each slice record contains everything the generator needs:
      - where to find the audio (path + start/end time)
      - timbre fingerprint (MFCC) for picking similar-sounding slices
      - harmonic fingerprint (chroma) for picking musically compatible slices
      - source tempo so we know how much to stretch
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # HPSS: cleaner beat tracking on percussive, chroma on harmonic
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Tempo + beat locations (from percussive)
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_percussive, sr=sr, hop_length=HOP_LENGTH
    )
    tempo = float(tempo) if hasattr(tempo, '__len__') == False else float(tempo[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    if len(beat_times) < 2:
        return []  # not enough beats to slice

    # Per-beat MFCC (timbre) -- synced to beat intervals from raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
    beat_mfcc = librosa.util.sync(mfcc, beat_frames)  # (N_MFCC, n_beats)

    # Per-beat chroma (harmonic/key) -- synced from harmonic signal
    chromagram = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=HOP_LENGTH
    )
    beat_chroma = librosa.util.sync(
        chromagram, beat_frames, aggregate=np.median
    )  # (12, n_beats)

    # Build one record per beat interval
    slices = []
    n_beats = min(beat_mfcc.shape[1], beat_chroma.shape[1], len(beat_times) - 1)
    for i in range(n_beats):
        start_time = beat_times[i]
        end_time = beat_times[i + 1] if i + 1 < len(beat_times) else duration
        slices.append({
            "path":       path,
            "beat_idx":   i,
            "start_time": start_time,
            "end_time":   end_time,
            "tempo":      tempo,
            "sr":         sr,
            "mfcc":       beat_mfcc[:, i],   # shape (13,)
            "chroma":     beat_chroma[:, i],  # shape (12,)
        })

    return slices


def main():
    parser = argparse.ArgumentParser(description="Analyze audio library and save index.")
    parser.add_argument("folder", help="Path to your music folder")
    args = parser.parse_args()

    files = find_audio_files(args.folder)
    if not files:
        print(f"No audio files found in {args.folder}")
        sys.exit(1)

    print(f"Found {len(files)} audio files. Analyzing...\n")

    all_slices = []
    for path in files:
        try:
            slices = analyze_track(path)
            all_slices.extend(slices)
            print(f"  {os.path.basename(path):45s} {len(slices):4d} beat-slices  "
                  f"BPM: {slices[0]['tempo']:.1f}" if slices else
                  f"  {os.path.basename(path):45s}  skipped (too short)")
        except Exception as e:
            print(f"  SKIPPED {os.path.basename(path)}: {e}")

    if not all_slices:
        print("\nNo slices could be extracted. Check your audio files.")
        sys.exit(1)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(all_slices, f)

    print(f"\nDone. {len(all_slices)} total beat-slices indexed from {len(files)} files.")
    print(f"Saved to: {INDEX_FILE}")
    print(f"\nNext step: python generate.py --bpm 120 --bars 8 --mode melodic")


if __name__ == "__main__":
    main()