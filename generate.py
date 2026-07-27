"""
Phase 2: Generate loopable audio from your analyzed library.

USAGE:
    python generate.py --bpm 120 --bars 8 --mode melodic

MODES:
    random    -- picks beat-slices randomly from your whole library
    melodic   -- picks slices that are harmonically compatible with each other
                 (similar chroma/key profile), creating something more musical
    tempo     -- picks slices from tracks closest to your target BPM,
                 minimizing the amount of time-stretching needed

OPTIONS:
    --bpm     Target tempo for the output loop (default: 120)
    --bars    How many bars long the output should be (default: 8)
    --beats   Beats per bar (default: 4, i.e. 4/4 time)
    --mode    random | melodic | tempo (default: melodic)
    --seed    Random seed for reproducible results (default: random)
    --index   Path to library_index.pkl (default: library_index.pkl)
    --out     Output wav filename (default: loop_<mode>.wav)

SETUP:
    Run analyze_and_slice.py first to generate library_index.pkl.
    pip install librosa soundfile numpy
"""

import os
import sys
import argparse
import pickle
import random
import numpy as np
import librosa
import soundfile as sf

INDEX_FILE = "library_index.pkl"
OUTPUT_DIR = "output"
HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """How similar are two feature vectors? 1.0 = identical, 0.0 = unrelated."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def most_compatible_slices(seed_slice, candidates, n, feature="chroma"):
    """Return the n slices from candidates most similar to seed_slice
    by cosine similarity on the given feature ('chroma' or 'mfcc').
    """
    seed_vec = seed_slice[feature]
    scored = [
        (cosine_similarity(seed_vec, s[feature]), s)
        for s in candidates
        if s is not seed_slice
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n]]


# ---------------------------------------------------------------------------
# Slice loading
# ---------------------------------------------------------------------------

# Cache audio files in memory so we don't reload the same file for every beat
_audio_cache = {}

def load_audio(path, sr):
    if path not in _audio_cache:
        y, _ = librosa.load(path, sr=sr, mono=True)
        _audio_cache[path] = y
    return _audio_cache[path]


def extract_slice(record):
    """Load the audio for one slice record and return the raw samples."""
    y = load_audio(record["path"], record["sr"])
    start = int(record["start_time"] * record["sr"])
    end   = int(record["end_time"]   * record["sr"])
    chunk = y[start:end]
    return chunk, record["sr"]


def stretch_to_bpm(chunk, sr, source_bpm, target_bpm):
    """Time-stretch chunk from source_bpm to target_bpm (pitch stays the same)."""
    if source_bpm <= 0 or abs(source_bpm - target_bpm) < 0.5:
        return chunk  # close enough, skip stretching
    rate = target_bpm / source_bpm
    return librosa.effects.time_stretch(chunk, rate=rate, hop_length=HOP_LENGTH)


# ---------------------------------------------------------------------------
# Selection modes
# ---------------------------------------------------------------------------

def select_random(library, n_beats):
    """Pick n_beats slices at random from the whole library."""
    return random.choices(library, k=n_beats)


def select_melodic(library, n_beats):
    """Pick a random seed slice, then fill the loop with harmonically
    compatible slices (similar chroma = similar key/pitch content).
    The first beat is the seed; the rest are the closest matches to it.
    """
    seed = random.choice(library)
    compatible = most_compatible_slices(seed, library, n=n_beats - 1, feature="chroma")

    # If library is smaller than n_beats, allow repeats
    while len(compatible) < n_beats - 1:
        compatible += compatible

    result = [seed] + compatible[:n_beats - 1]
    random.shuffle(result)  # shuffle so it doesn't just ramp from most→least similar
    return result


def select_tempo(library, n_beats, target_bpm):
    """Pick slices from tracks whose native tempo is closest to the target,
    so time-stretching is minimal and artifacts are less likely.
    """
    library_sorted = sorted(library, key=lambda s: abs(s["tempo"] - target_bpm))
    # Take the top 25% closest-tempo slices, then pick randomly from those
    pool_size = max(n_beats, len(library_sorted) // 4)
    pool = library_sorted[:pool_size]
    return random.choices(pool, k=n_beats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a loop from your music library.")
    parser.add_argument("--bpm",   type=float, default=120.0, help="Target tempo (default: 120)")
    parser.add_argument("--bars",  type=int,   default=8,     help="Loop length in bars (default: 8)")
    parser.add_argument("--beats", type=int,   default=4,     help="Beats per bar (default: 4)")
    parser.add_argument("--mode",  type=str,   default="melodic",
                        choices=["random", "melodic", "tempo"],
                        help="Slice selection mode (default: melodic)")
    parser.add_argument("--seed",  type=int,   default=None,  help="Random seed for reproducibility")
    parser.add_argument("--index", type=str,   default=INDEX_FILE, help="Path to library index")
    parser.add_argument("--out",   type=str,   default=None,  help="Output filename")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load library index
    if not os.path.exists(args.index):
        print(f"Index file not found: {args.index}")
        print("Run analyze_and_slice.py first.")
        sys.exit(1)

    with open(args.index, "rb") as f:
        library = pickle.load(f)

    print(f"Loaded {len(library)} beat-slices from index.")

    # How many beat-slices do we need?
    n_beats = args.bars * args.beats
    print(f"Generating {args.bars}-bar loop at {args.bpm} BPM "
          f"({n_beats} beats, mode: {args.mode})\n")

    # Select slices
    if args.mode == "random":
        selected = select_random(library, n_beats)
    elif args.mode == "melodic":
        selected = select_melodic(library, n_beats)
    elif args.mode == "tempo":
        selected = select_tempo(library, n_beats, args.bpm)

    # Load, stretch, and stitch each slice
    beat_audio = []
    output_sr = selected[0]["sr"]  # use first slice's sample rate as master

    for i, record in enumerate(selected):
        chunk, sr = extract_slice(record)
        stretched  = stretch_to_bpm(chunk, sr, record["tempo"], args.bpm)

        # Resample to master sr if tracks have mixed sample rates
        if sr != output_sr:
            stretched = librosa.resample(stretched, orig_sr=sr, target_sr=output_sr)

        beat_audio.append(stretched)
        print(f"  Beat {i+1:3d}/{n_beats}  "
              f"{os.path.basename(record['path']):40s}  "
              f"beat {record['beat_idx']:3d}  "
              f"src BPM: {record['tempo']:.1f}")

    # Concatenate all beats into one loop
    loop = np.concatenate(beat_audio)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_name = args.out or f"loop_{args.mode}_{int(args.bpm)}bpm_{args.bars}bars.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, loop, output_sr)

    duration = len(loop) / output_sr
    print(f"\nSaved: {out_path}  ({duration:.1f}s)")


if __name__ == "__main__":
    main()