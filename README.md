# Sparta Remix Generator

Generate a “Sparta Remix” (audio + aligned video) from an input video by automatically extracting samples, pitch-correcting them, arranging them into a Sparta-style structure, and rendering a final MP4.

This repo contains:

- A **CLI** generator (`main.py`)
- A **Windows GUI** (CustomTkinter) (`gui.py`)
- Audio sample extraction / analysis (`audio_processor.py`)
- Pattern + song structure generation (`pattern_generator.py`)
- Arrangement, mixing, and video rendering (`arranger.py`)

## Features

- Extracts audio from an input video and slices it into categorized samples (vocals, drums, bass, etc.)
- Picks “clean” vocal samples for chorus (pure vowel selection)
- Pitch-corrects different “instruments” (pitch, melody, bass, pads) and places them into the arrangement
- Generates a Sparta-style song structure with selectable length
- Renders:
  - `output/sparta_remix.wav`
  - `output/sparta_remix.mp4` (video is cut/composited to align with the generated hits)
- GUI with:
  - Input preview
  - Quote start/end selection
  - Remix length slider
  - Instrument toggles
  - Output preview/play button

## Requirements

- **Windows** (recommended; project includes Windows binaries)
- **Python 3.10+** (3.11 is fine too)
- **FFmpeg**
  - The app attempts to use a bundled `ffmpeg.exe` if present or an ImageIO-provided FFmpeg.
- Included binaries used by the pipeline:
  - `rubberband.exe` + `sndfile.dll` (time-stretching / pitch-related)
  - `PitchCorrector297_V1.0/.../PitchCorrector297.exe` (used for specific pitch workflows)

## Install

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage (CLI)

```bash
python main.py path\to\input_video.mp4 -o output\sparta_remix.mp4 --bpm 140 --length 6
```

Arguments:

- `input_video` (required): Path to input video
- `-o/--output`: Output path. If you pass an `.mp4`, audio will also be written as a `.wav` next to it.
- `--bpm`: Tempo (default `140`)
- `--length`: Remix length (1–6)
  - `1` = Intro only
  - `6` = Full remix

## Usage (GUI)

```bash
python gui.py
```

Then:

- Choose an input video
- (Optional) adjust quote start/end (used to bias chorus vocal selection)
- Choose remix length
- Toggle instruments
- Click **Generate Sparta Remix**

## Outputs & folders

- `output/` and `temp/` are created automatically (see `config.py`).
- Final outputs default to:
  - `output/sparta_remix.wav`
  - `output/sparta_remix.mp4`

## Packaging (PyInstaller)

A PyInstaller spec is included: `sparta_gui.spec`.

Build:

```bash
pyinstaller sparta_gui.spec
```

The spec bundles required runtime files like:

- `rubberband.exe`
- `sndfile.dll`
- `build/gun/GUN.wav`
- `base.wav`
- `icon.png`

## Notes about Git tracking

This repo’s `.gitignore` is set up to avoid committing large/generated files:

- `output/`, `temp/`, `dist/`
- `*.mp4`, `*.wav` (except `base.wav` and `build/gun/GUN.wav`)

If you add new required runtime media, update `.gitignore` accordingly.

## Troubleshooting

- If video rendering fails, verify FFmpeg is available on PATH or that a valid `ffmpeg.exe` is discoverable.
- If time-stretch/pitch steps fail, verify `rubberband.exe` and `sndfile.dll` exist in the project root (or are correctly bundled in your packaged build).
- If you run the packaged EXE and it “does nothing”, check `sparta_gui.log` (the GUI redirects stdout/stderr when frozen).

## Credit

If used and published on social media, please credit:

- YouTube: `@krasen671`
- Or link to this GitHub repo
