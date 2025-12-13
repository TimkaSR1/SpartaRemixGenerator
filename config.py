import os
import sys
from pathlib import Path

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

# Project directories
def get_resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass_dir = Path(getattr(sys, "_MEIPASS", exe_dir)).resolve()

        if (exe_dir / "build").exists() or (exe_dir / "icon.png").exists() or (exe_dir / "base.wav").exists():
            return exe_dir
        return meipass_dir

    return Path(__file__).parent.resolve()

def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if os.access(str(exe_dir), os.W_OK):
            return exe_dir
        local = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))).resolve()
        return local / "SpartaRemixGenerator"
    return Path(__file__).parent.resolve()

RESOURCE_DIR = get_resource_dir()
BASE_DIR = get_app_dir()

_bundled_ffmpeg = RESOURCE_DIR / "ffmpeg.exe"
if not _bundled_ffmpeg.exists():
    _candidates = sorted(RESOURCE_DIR.glob("ffmpeg*.exe"))
    if _candidates:
        _bundled_ffmpeg = _candidates[0]

if _bundled_ffmpeg.exists():
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", str(_bundled_ffmpeg))
    os.environ.setdefault("FFMPEG_BINARY", str(_bundled_ffmpeg))
elif imageio_ffmpeg is not None:
    try:
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
SAMPLE_DIR = TEMP_DIR / "samples"

# Audio settings
SAMPLE_RATE = 44100
BIT_DEPTH = 16
CHANNELS = 2

# Sample settings
MIN_SAMPLE_LENGTH = 0.1  # seconds
MAX_SAMPLE_LENGTH = 2.0  # seconds
SILENCE_THRESHOLD = -40  # dB

# Sparta structure settings
BPM = 140
BEAT_DURATION = 60 / BPM  # seconds
BAR_DURATION = BEAT_DURATION * 4  # 4/4 time

# Pattern settings
PATTERNS = {
    'intro': {
        'length': 4,  # bars
        'drums': "K...H...K...H...",  # K=kick, H=hihat, S=snare, .=rest
        'bass': "1...2...3...4...",
        'chords': ["C3", "G3", "A3", "F3"],
    },
    'chorus': {
        'length': 8,
        'drums': "K...S...K...S...K...S...K...S...",
        'bass': "1_1_2_2_3_3_4_4_",
        'chords': ["C3", "G3", "A3", "F3"] * 2,
        'vocals': "11_11_111_1_1_11222_2_222_222_2_"
    },
    'madness': {
        'length': 16,
        'drums': "KHS.KHS.KHS.KHS.",
        'bass': '1_1_1_1_2_2_2_2_3_3_3_3_4_4_4_4_',
        'chords': ["C3", "G3", "A3", "F3"] * 4,
        'vocals': '1234123412341234'
    },
    'epicness': {
        'length': 8,
        'drums': "K.......S.......K.......S.......",
        'bass': '1.......2.......3.......4.......',
        'chords': ["C3", "G3", "A3", "F3"] * 2,
        'vocals': '1...2...3...4...1...2...3...4...'
    }
}

# Sample categories
SAMPLE_CATEGORIES = {
    'kick': {
        'freq_range': (60, 120),  # Hz
        'max_duration': 0.3,  # seconds
        'min_energy': 0.5  # 0-1 scale
    },
    'snare': {
        'freq_range': (150, 300),
        'max_duration': 0.4,
        'min_energy': 0.4
    },
    'hihat': {
        'freq_range': (1000, 10000),
        'max_duration': 0.2,
        'min_energy': 0.3
    },
    'bass': {
        'freq_range': (30, 200),
        'max_duration': 1.0,
        'min_energy': 0.4
    },
    'chord': {
        'freq_range': (100, 2000),
        'max_duration': 1.5,
        'min_energy': 0.3
    },
    'vocal': {
        'freq_range': (200, 3000),
        'max_duration': 1.0,
        'min_energy': 0.2
    }
}

def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [OUTPUT_DIR, TEMP_DIR, SAMPLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
