import os
import sys
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import tempfile
import shutil
import json
from scipy import signal
from scipy.ndimage import uniform_filter1d
import math

# Try to import pyrubberband for high-quality pitch-preserving time stretch
# Also ensure rubberband CLI is findable by adding project dir to PATH
try:
    # Add project directory to PATH so rubberband.exe can be found
    _project_dir = (
        str(Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve())
        if getattr(sys, "frozen", False)
        else os.path.dirname(os.path.abspath(__file__))
    )
    _rubberband_exe = os.path.join(_project_dir, "rubberband.exe")
    if os.path.exists(_rubberband_exe):
        os.environ["PATH"] = _project_dir + os.pathsep + os.environ.get("PATH", "")

    import pyrubberband as pyrb
    HAS_PYRUBBERBAND = True
except ImportError:
    HAS_PYRUBBERBAND = False
    print("[Warning] pyrubberband not installed. Using librosa for time stretching (lower quality).")
    print("          Install with: pip install pyrubberband")
    print("          Also requires rubberband CLI: https://breakfastquay.com/rubberband/")

from config import (
    SAMPLE_RATE, TEMP_DIR, SAMPLE_DIR, MIN_SAMPLE_LENGTH, 
    MAX_SAMPLE_LENGTH, SILENCE_THRESHOLD, SAMPLE_CATEGORIES, BASE_DIR, RESOURCE_DIR
)

# Path to PitchCorrector297 CLI executable
PITCH_CORRECTOR_EXE = RESOURCE_DIR / "PitchCorrector297_V1.0" / "PitchCorrector297" / "bin" / "Debug" / "PitchCorrector297.exe"


class SpeechDetectorConfig:
    """
    Tunable configuration for speech detection.
    Adjust these parameters to fine-tune detection for different audio sources.
    """
    def __init__(self):
        # Speech detection thresholds (LOWER = more permissive)
        self.speech_threshold = 0.25  # Min score to consider as speech (0-1)
        self.min_region_ms = 50       # Minimum speech region duration in ms
        
        # Energy detection
        self.energy_percentile = 20   # Percentile for dynamic energy threshold
        self.energy_weight = 0.35     # Weight for energy in speech score
        
        # Zero-crossing rate (speech range)
        self.zcr_min = 0.01           # Min ZCR for speech
        self.zcr_max = 0.25           # Max ZCR for speech (higher = more permissive)
        self.zcr_weight = 0.15        # Weight for ZCR in speech score
        
        # Spectral flatness (lower = more tonal/speech-like)
        self.flatness_max = 0.5       # Max flatness for speech
        self.flatness_weight = 0.15   # Weight for flatness in speech score
        
        # Pitch detection
        self.pitch_mag_threshold = 0.02  # Min magnitude for pitch detection (LOWERED)
        self.pitch_min = 60           # Min fundamental frequency (Hz)
        self.pitch_max = 500          # Max fundamental frequency (Hz) - wider range
        self.pitch_weight = 0.25      # Weight for pitch in speech score
        
        # Harmonic content
        self.hnr_threshold = 0.8      # Min harmonic-to-noise ratio
        self.hnr_weight = 0.1         # Weight for HNR in speech score
        
        # Phoneme analysis - penalties (REDUCED for more permissive detection)
        self.sibilant_penalty = 25    # Penalty for SSS sounds (was 50)
        self.fricative_penalty = 15   # Penalty for F/V/TH sounds (was 30)
        self.plosive_penalty = 10     # Penalty for T/D/K sounds (was 25)
        
        # Phoneme analysis - bonuses
        self.loudness_bonus_max = 40  # Max bonus for loudness
        self.vowel_bonus = 50         # Bonus for vowel content
        self.voiced_bonus = 25        # Bonus for voiced content
        
        # Quality thresholds
        self.min_quality = 10.0       # Minimum quality score to accept (LOWERED)
        self.good_quality = 30.0      # Quality score considered "good"
    
    def save(self, path: str):
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def load(self, path: str):
        """Load config from JSON file."""
        import json
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(self, k):
                        setattr(self, k, v)
    
    @classmethod
    def permissive(cls) -> 'SpeechDetectorConfig':
        """Create a very permissive config that detects almost any vocal sound."""
        cfg = cls()
        cfg.speech_threshold = 0.15
        cfg.min_region_ms = 30
        cfg.energy_percentile = 10
        cfg.zcr_max = 0.35
        cfg.flatness_max = 0.7
        cfg.pitch_mag_threshold = 0.01
        cfg.hnr_threshold = 0.5
        cfg.sibilant_penalty = 10
        cfg.fricative_penalty = 5
        cfg.plosive_penalty = 5
        cfg.min_quality = 5.0
        return cfg
    
    @classmethod
    def strict(cls) -> 'SpeechDetectorConfig':
        """Create a strict config that only detects clear speech."""
        cfg = cls()
        cfg.speech_threshold = 0.4
        cfg.min_region_ms = 100
        cfg.energy_percentile = 30
        cfg.zcr_max = 0.15
        cfg.flatness_max = 0.3
        cfg.pitch_mag_threshold = 0.05
        cfg.hnr_threshold = 1.5
        cfg.sibilant_penalty = 50
        cfg.fricative_penalty = 30
        cfg.plosive_penalty = 25
        cfg.min_quality = 25.0
        return cfg
    
    @classmethod
    def create_tuning_file(cls, path: str = "speech_config.json"):
        """Create a tunable config file with comments explaining each parameter."""
        cfg = cls.permissive()  # Start with permissive defaults
        cfg.save(path)
        print(f"Created tuning config at: {path}")
        print("Edit this file to adjust speech detection parameters.")
        print("\nKey parameters to tune:")
        print("  - speech_threshold: LOWER = detect more speech (0.1-0.5)")
        print("  - min_quality: LOWER = accept more samples (5-30)")
        print("  - sibilant_penalty: LOWER = allow more 'sss' sounds (5-50)")
        print("  - vowel_bonus: HIGHER = prefer clear vowels (30-60)")
        return path
    
    def adjust_for_feedback(self, detected_speech: bool, was_correct: bool):
        """
        Simple learning: adjust parameters based on feedback.
        Call this after each detection to improve over time.
        
        Args:
            detected_speech: Did the algorithm detect speech?
            was_correct: Was the detection correct?
        """
        if detected_speech and not was_correct:
            # False positive - tighten thresholds
            self.speech_threshold = min(0.6, self.speech_threshold + 0.02)
            self.min_quality = min(40, self.min_quality + 2)
        elif not detected_speech and not was_correct:
            # False negative - loosen thresholds
            self.speech_threshold = max(0.1, self.speech_threshold - 0.03)
            self.min_quality = max(3, self.min_quality - 3)
            self.energy_percentile = max(5, self.energy_percentile - 2)


class SpeechDetector:
    """
    Smart speech detection and vocal peak extraction.
    Detects actual human speech, finds transient peaks, and filters sibilants/fricatives.
    
    Use SpeechDetectorConfig to tune detection parameters.
    """
    
    # Frequency bands for phoneme classification
    SIBILANT_BAND = (4000, 10000)  # SSS, SH, Z sounds
    FRICATIVE_BAND = (2500, 8000)  # F, V, TH sounds  
    PLOSIVE_BAND = (1000, 4000)    # T, D, K, P, B sounds (burst)
    VOWEL_FORMANT_BAND = (300, 3000)  # Vowel formants F1-F3
    
    def __init__(self, sr: int = 44100, config: SpeechDetectorConfig = None):
        self.sr = sr
        self.config = config or SpeechDetectorConfig()
        
    def detect_speech_regions(self, y: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Detect regions containing human speech using multiple features.
        Returns list of (start_sample, end_sample, speech_confidence) tuples.
        """
        cfg = self.config
        hop_length = 512
        frame_length = 2048
        
        # 1. Energy-based VAD
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        rms_threshold = np.percentile(rms_db, cfg.energy_percentile)
        
        # 2. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 3. Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        
        # 4. Pitch presence - with more lenient detection
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr, hop_length=hop_length)
        pitch_presence = np.zeros(pitches.shape[1])
        for t in range(pitches.shape[1]):
            valid_pitches = pitches[:, t][(magnitudes[:, t] > cfg.pitch_mag_threshold) & 
                                          (pitches[:, t] >= cfg.pitch_min) &
                                          (pitches[:, t] <= cfg.pitch_max)]
            pitch_presence[t] = len(valid_pitches) > 0
        
        # 5. Harmonic-to-noise ratio approximation
        try:
            harm, perc = librosa.effects.hpss(y)
            harm_rms = librosa.feature.rms(y=harm, hop_length=hop_length)[0]
            perc_rms = librosa.feature.rms(y=perc, hop_length=hop_length)[0]
            hnr = harm_rms / (perc_rms + 1e-10)
        except:
            hnr = np.ones_like(rms)
        
        # Combine features into speech probability per frame
        min_len = min(len(rms), len(zcr), len(flatness), len(pitch_presence), len(hnr))
        speech_prob = np.zeros(min_len)
        
        for i in range(min_len):
            score = 0.0
            # Energy above threshold
            if rms_db[i] > rms_threshold:
                score += cfg.energy_weight
            # ZCR in speech range
            if cfg.zcr_min < zcr[i] < cfg.zcr_max:
                score += cfg.zcr_weight
            # Low spectral flatness (tonal)
            if flatness[i] < cfg.flatness_max:
                score += cfg.flatness_weight
            # Pitch present
            if pitch_presence[i]:
                score += cfg.pitch_weight
            # Good harmonic content
            if hnr[i] > cfg.hnr_threshold:
                score += cfg.hnr_weight
            
            speech_prob[i] = score
        
        # Smooth the probability
        speech_prob = uniform_filter1d(speech_prob, size=5)
        
        # Find contiguous speech regions
        is_speech = speech_prob > cfg.speech_threshold
        
        regions = []
        in_region = False
        start_frame = 0
        min_samples = int(cfg.min_region_ms / 1000 * self.sr)
        
        for i, is_sp in enumerate(is_speech):
            if is_sp and not in_region:
                start_frame = i
                in_region = True
            elif not is_sp and in_region:
                end_frame = i
                start_sample = start_frame * hop_length
                end_sample = min(end_frame * hop_length, len(y))
                avg_conf = np.mean(speech_prob[start_frame:end_frame])
                if end_sample - start_sample > min_samples:
                    regions.append((start_sample, end_sample, avg_conf))
                in_region = False
        
        # Handle region at end
        if in_region:
            end_sample = len(y)
            avg_conf = np.mean(speech_prob[start_frame:])
            if end_sample - start_frame * hop_length > min_samples:
                regions.append((start_frame * hop_length, end_sample, avg_conf))
        
        # FALLBACK: If no regions detected, find the loudest segments instead
        if not regions:
            print("    [SpeechDetector] No speech regions found, using loudest segments as fallback")
            regions = self._find_loudest_segments(y, rms, hop_length, top_n=3)
        
        return regions
    
    def _find_loudest_segments(self, y: np.ndarray, rms: np.ndarray, hop_length: int, 
                                top_n: int = 3) -> List[Tuple[int, int, float]]:
        """Fallback: find the loudest segments when speech detection fails."""
        # Find peaks in RMS
        rms_smooth = uniform_filter1d(rms, size=10)
        threshold = np.percentile(rms_smooth, 60)
        
        # Find segments above threshold
        above_threshold = rms_smooth > threshold
        regions = []
        in_region = False
        start_frame = 0
        
        for i, above in enumerate(above_threshold):
            if above and not in_region:
                start_frame = i
                in_region = True
            elif not above and in_region:
                end_frame = i
                start_sample = start_frame * hop_length
                end_sample = min(end_frame * hop_length, len(y))
                avg_rms = np.mean(rms_smooth[start_frame:end_frame])
                if end_sample - start_sample > int(0.05 * self.sr):  # Min 50ms
                    regions.append((start_sample, end_sample, float(avg_rms)))
                in_region = False
        
        if in_region:
            end_sample = len(y)
            avg_rms = np.mean(rms_smooth[start_frame:])
            if end_sample - start_frame * hop_length > int(0.05 * self.sr):
                regions.append((start_frame * hop_length, end_sample, float(avg_rms)))
        
        # Sort by loudness and return top N
        regions.sort(key=lambda x: x[2], reverse=True)
        return regions[:top_n]
    
    def analyze_phoneme_type(self, y: np.ndarray) -> Dict[str, float]:
        """
        Analyze audio segment for phoneme characteristics.
        Returns scores for different phoneme types.
        """
        # Compute spectrogram
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        
        # Energy in different frequency bands
        def band_energy(low, high):
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                return np.mean(S[mask, :])
            return 0.0
        
        total_energy = np.mean(S) + 1e-10
        
        # Sibilant energy (SSS, SH, Z) - very high frequency hiss
        sibilant_energy = band_energy(*self.SIBILANT_BAND)
        sibilant_ratio = sibilant_energy / total_energy
        
        # Fricative energy (F, V, TH) - mid-high frequency
        fricative_energy = band_energy(*self.FRICATIVE_BAND)
        fricative_ratio = fricative_energy / total_energy
        
        # Vowel formant energy
        vowel_energy = band_energy(*self.VOWEL_FORMANT_BAND)
        vowel_ratio = vowel_energy / total_energy
        
        # Plosive detection (T, D, K, P, B) - sharp transient at start
        onset_ms = 30  # First 30ms
        onset_samples = int(onset_ms / 1000 * self.sr)
        
        if len(y) > onset_samples * 3:
            onset_rms = np.sqrt(np.mean(y[:onset_samples]**2))
            body_rms = np.sqrt(np.mean(y[onset_samples:]**2)) + 1e-10
            plosive_ratio = onset_rms / body_rms
            
            # Check for silence after burst (characteristic of plosives)
            post_onset = y[onset_samples:onset_samples*2] if len(y) > onset_samples*2 else y[onset_samples:]
            post_onset_rms = np.sqrt(np.mean(post_onset**2))
            has_gap = post_onset_rms < onset_rms * 0.3
        else:
            plosive_ratio = 0.0
            has_gap = False
        
        # Voiced sound detection (vowels, voiced consonants)
        try:
            harm, perc = librosa.effects.hpss(y)
            harm_ratio = np.sqrt(np.mean(harm**2)) / (np.sqrt(np.mean(y**2)) + 1e-10)
        except:
            harm_ratio = 0.5
        
        return {
            'sibilant': min(1.0, sibilant_ratio * 3),  # SSS, SH, Z
            'fricative': min(1.0, fricative_ratio * 2),  # F, V, TH
            'plosive': min(1.0, plosive_ratio - 0.5) if plosive_ratio > 1.0 else 0.0,  # T, D, K, P
            'vowel': min(1.0, vowel_ratio * 1.5) * harm_ratio,  # Clear vowels
            'voiced': harm_ratio,  # General voiced content
            'has_plosive_gap': has_gap,
        }
    
    def find_vocal_peaks(self, y: np.ndarray, top_n: int = 5) -> List[Dict]:
        """
        Find the best transient peaks in the audio that are vowel-heavy.
        These are the loud, clear parts of speech - not the sibilants/fricatives.
        
        Returns list of dicts with peak info, sorted by quality.
        """
        cfg = self.config
        hop_length = 512
        
        # Get onset envelope for transient detection
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=hop_length)
        
        # Find peaks in onset envelope - use lower delta for more peaks
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, 
                                        pre_avg=3, post_avg=5, delta=0.05, wait=5)
        
        # If no peaks found with peak_pick, find top N loudest frames
        if len(peaks) == 0:
            # Fallback: find frames with highest onset strength
            sorted_frames = np.argsort(onset_env)[::-1]
            peaks = sorted_frames[:top_n * 2]
        
        # Analyze each peak region
        peak_info = []
        window_before = int(0.02 * self.sr)  # 20ms before peak
        window_after = int(0.15 * self.sr)   # 150ms after peak (capture vowel)
        
        for peak_frame in peaks:
            peak_sample = peak_frame * hop_length
            start = max(0, peak_sample - window_before)
            end = min(len(y), peak_sample + window_after)
            
            if end - start < int(0.03 * self.sr):  # Min 30ms (reduced from 50ms)
                continue
            
            segment = y[start:end]
            
            # Analyze phoneme type
            phoneme = self.analyze_phoneme_type(segment)
            
            # Calculate peak quality score using config parameters
            rms = np.sqrt(np.mean(segment**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Quality score calculation
            quality = 0.0
            
            # Loudness bonus
            quality += min(cfg.loudness_bonus_max, max(0, (rms_db + 30) * 1.5))
            
            # Vowel content bonus - THIS IS WHAT WE WANT
            quality += phoneme['vowel'] * cfg.vowel_bonus
            
            # Voiced content bonus
            quality += phoneme['voiced'] * cfg.voiced_bonus
            
            # PENALTIES for unwanted sounds (using config values)
            quality -= phoneme['sibilant'] * cfg.sibilant_penalty
            quality -= phoneme['fricative'] * cfg.fricative_penalty
            quality -= phoneme['plosive'] * cfg.plosive_penalty
            
            peak_info.append({
                'start_sample': start,
                'end_sample': end,
                'peak_sample': peak_sample,
                'rms': rms,
                'rms_db': rms_db,
                'phoneme': phoneme,
                'quality': max(0, quality),
            })
        
        # Sort by quality descending
        peak_info.sort(key=lambda x: x['quality'], reverse=True)
        
        return peak_info[:top_n]
    
    def extract_best_vocal_hits(self, y: np.ndarray, num_hits: int = 2,
                                 min_separation_ms: float = 150) -> List[Dict]:
        """
        Extract the best vocal hits from audio for use in Sparta remix.
        Prioritizes clear, loud vowel sounds and ignores sibilants/fricatives.
        
        Args:
            y: Audio signal
            num_hits: Number of vocal hits to extract
            min_separation_ms: Minimum time between hits in milliseconds
            
        Returns:
            List of dicts with 'audio', 'start', 'end', 'quality' info
        """
        # First check if there's any speech at all
        speech_regions = self.detect_speech_regions(y)
        
        if not speech_regions:
            print("  [SpeechDetector] No clear speech detected, using fallback")
            # Fallback: just find loudest transients
            peaks = self.find_vocal_peaks(y, top_n=num_hits * 3)
            if not peaks:
                return []
            return self._extract_hits_from_peaks(y, peaks[:num_hits], min_separation_ms)
        
        print(f"  [SpeechDetector] Found {len(speech_regions)} speech regions")
        
        # Analyze peaks within speech regions
        all_peaks = []
        for start, end, conf in speech_regions:
            segment = y[start:end]
            peaks = self.find_vocal_peaks(segment, top_n=5)
            
            # Adjust peak positions to global coordinates
            for peak in peaks:
                peak['start_sample'] += start
                peak['end_sample'] += start
                peak['peak_sample'] += start
                peak['speech_confidence'] = conf
                # Boost quality by speech confidence
                peak['quality'] *= (0.5 + conf * 0.5)
            
            all_peaks.extend(peaks)
        
        # Sort all peaks by quality
        all_peaks.sort(key=lambda x: x['quality'], reverse=True)
        
        return self._extract_hits_from_peaks(y, all_peaks, min_separation_ms, num_hits)
    
    def _extract_hits_from_peaks(self, y: np.ndarray, peaks: List[Dict], 
                                  min_separation_ms: float, max_hits: int = 2) -> List[Dict]:
        """Extract non-overlapping hits from peak list."""
        min_sep_samples = int(min_separation_ms / 1000 * self.sr)
        
        selected = []
        for peak in peaks:
            if len(selected) >= max_hits:
                break
            
            # Check separation from already selected
            too_close = False
            for sel in selected:
                if abs(peak['peak_sample'] - sel['peak_sample']) < min_sep_samples:
                    too_close = True
                    break
            
            if not too_close:
                # Extract the actual audio
                start = peak['start_sample']
                end = peak['end_sample']
                audio = y[start:end].copy()
                
                # Apply short fade to avoid clicks
                fade_samples = min(int(0.005 * self.sr), len(audio) // 4)
                if fade_samples > 0:
                    audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                selected.append({
                    'audio': audio,
                    'start': start / self.sr,
                    'end': end / self.sr,
                    'quality': peak['quality'],
                    'phoneme': peak.get('phoneme', {}),
                    'rms_db': peak.get('rms_db', -20),
                })
                
                phoneme = peak.get('phoneme', {})
                print(f"  [SpeechDetector] Selected hit: {start/self.sr:.2f}s-{end/self.sr:.2f}s, "
                      f"quality={peak['quality']:.1f}, vowel={phoneme.get('vowel', 0):.2f}, "
                      f"sibilant={phoneme.get('sibilant', 0):.2f}")
        
        return selected
    
    def is_good_speech_sample(self, y: np.ndarray, min_quality: float = None) -> Tuple[bool, float, Dict]:
        """
        Check if an audio sample is good speech (not sibilant/fricative heavy).
        
        Returns:
            (is_good, quality_score, phoneme_analysis)
        """
        cfg = self.config
        if min_quality is None:
            min_quality = cfg.min_quality
            
        phoneme = self.analyze_phoneme_type(y)
        
        # Calculate quality using config parameters
        rms = np.sqrt(np.mean(y**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        quality = 0.0
        quality += min(cfg.loudness_bonus_max, max(0, (rms_db + 30) * 1.5))
        quality += phoneme['vowel'] * cfg.vowel_bonus
        quality += phoneme['voiced'] * cfg.voiced_bonus
        quality -= phoneme['sibilant'] * (cfg.sibilant_penalty * 1.5)
        quality -= phoneme['fricative'] * cfg.fricative_penalty
        quality -= phoneme['plosive'] * cfg.plosive_penalty
        
        # More permissive "good" check - just need to meet minimum quality
        is_good = quality >= min_quality
        
        return is_good, max(0, quality), phoneme


class AudioProcessor:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.sample_dir = SAMPLE_DIR
        # Default sample rate used across analysis helpers
        self.sample_rate = SAMPLE_RATE
    
    def time_stretch_preserve_pitch(self, y: np.ndarray, sr: int, rate: float, crisp: int = 6) -> np.ndarray:
        """
        Time-stretch audio while preserving pitch using the best available method.
        
        Uses pyrubberband (RubberBand library) if available for high-quality results,
        otherwise falls back to librosa's phase vocoder.
        
        Args:
            y: Audio time series (mono)
            sr: Sample rate
            rate: Stretch rate. rate > 1 speeds up (shorter), rate < 1 slows down (longer)
                  e.g., rate=0.5 makes audio 2x longer, rate=2.0 makes it 2x shorter
            crisp: Crispness level for RubberBand (0-6, default 6 for best transients)
        
        Returns:
            Time-stretched audio with original pitch preserved
        """
        if rate == 1.0:
            return y
        
        if HAS_PYRUBBERBAND:
            try:
                # pyrubberband provides high-quality pitch-preserving time stretch
                # using the RubberBand library with crisp setting for best quality
                y_stretched = pyrb.time_stretch(y, sr, rate, rbargs={'--crisp': str(crisp)})
                return y_stretched
            except Exception as e:
                print(f"  [TimeStretch] pyrubberband failed: {e}, falling back to librosa")
    
    def prepare_sample_standard(self, sample_path: str, output_path: str, 
                                 freq: float = 293.66, stretch_rate: float = 0.2,
                                 trim_start_percent: float = 0.08, 
                                 lowpass_freq: Optional[float] = None,
                                 label: str = "Sample") -> Optional[str]:
        """
        Standard sample preparation pipeline:
        1. PitchCorrector297 for pitch correction
        2. RubberBand time stretch with crisp=6
        3. Trim beginning of sample
        4. Optional lowpass filter
        5. Normalize and apply fades
        
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV
            freq: Target frequency in Hz (default 293.66 = D4)
            stretch_rate: Time stretch rate (0.2 = 5x longer)
            trim_start_percent: Trim this percent from start (0.08 = 8%)
            lowpass_freq: Optional lowpass filter cutoff in Hz
            label: Label for logging
            
        Returns:
            Path to prepared sample, or None on failure
        """
        import subprocess
        
        if not PITCH_CORRECTOR_EXE.exists():
            print(f"  [{label}] PitchCorrector297.exe not found")
            return None
        
        try:
            # Step 1: Run PitchCorrector297 for pitch correction
            temp_pitched = str(self.temp_dir / f"{label.lower()}_pitched_temp.wav")
            result = subprocess.run(
                [str(PITCH_CORRECTOR_EXE), sample_path, temp_pitched, str(freq)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 or not os.path.exists(temp_pitched):
                print(f"  [{label}] PitchCorrector297 failed: {result.stderr or result.stdout}")
                return None
            
            print(f"  [{label}] Pitch corrected to {freq:.1f} Hz")
            
            # Step 2: Load and time stretch with RubberBand crisp=6
            y, sr = librosa.load(temp_pitched, sr=SAMPLE_RATE, mono=True)
            
            print(f"  [{label}] Time stretching with rate={stretch_rate} (making {1/stretch_rate:.1f}x longer)")
            y_stretched = self.time_stretch_preserve_pitch(y, sr, stretch_rate, crisp=6)
            
            # Step 3: Trim beginning
            trim_samples = int(len(y_stretched) * trim_start_percent)
            y_trimmed = y_stretched[trim_samples:]
            print(f"  [{label}] Trimmed {trim_start_percent*100:.0f}% from start")
            
            # Step 4: Optional lowpass filter
            if lowpass_freq is not None:
                nyquist = sr / 2
                normalized_cutoff = lowpass_freq / nyquist
                normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
                sos = signal.butter(2, normalized_cutoff, btype='low', output='sos')
                y_trimmed = signal.sosfilt(sos, y_trimmed)
                print(f"  [{label}] Applied lowpass filter at {lowpass_freq} Hz")
            
            # Step 5: Normalize
            y_final = librosa.util.normalize(y_trimmed)
            
            # Step 6: Apply fades
            fade_in_samples = int(0.02 * sr)  # 20ms fade in
            fade_out_samples = int(0.05 * sr)  # 50ms fade out
            
            if len(y_final) > fade_in_samples + fade_out_samples:
                y_final[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
                y_final[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
            
            # Save
            sf.write(output_path, y_final, sr)
            print(f"  [{label}] Prepared: {output_path} ({len(y_final)/sr:.2f}s)")
            
            # Clean up temp file
            if os.path.exists(temp_pitched):
                os.remove(temp_pitched)
            
            return output_path
            
        except subprocess.TimeoutExpired:
            print(f"  [{label}] PitchCorrector297 timed out")
            return None
        except Exception as e:
            print(f"  [{label}] Failed to prepare sample: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None,
                                  start_time: Optional[float] = None, 
                                  end_time: Optional[float] = None) -> str:
        """
        Extract audio from video file using pydub.
        
        Args:
            video_path: Path to video file
            output_path: Output WAV path
            start_time: Start time in seconds (optional, for quote selection)
            end_time: End time in seconds (optional, for quote selection)
        """
        if output_path is None:
            output_path = str(self.temp_dir / "extracted_audio.wav")
        
        ffmpeg_exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            AudioSegment.converter = ffmpeg_exe
        
        video = AudioSegment.from_file(video_path)
        
        # Trim to quote selection if specified (and valid range)
        if start_time is not None and end_time is not None and end_time > start_time:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            if end_ms > start_ms and end_ms <= len(video):
                video = video[start_ms:end_ms]
                print(f"Trimmed audio to {start_time:.2f}s - {end_time:.2f}s ({len(video)}ms)")
            else:
                print(f"Quote range invalid, using full audio ({len(video)}ms)")
        
        video.export(output_path, format="wav")
        return output_path
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa."""
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return y, sr
    
    def detect_onsets(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Detect onsets in the audio signal."""
        # Use librosa's onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, 
            sr=sr, 
            units='time',
            hop_length=512,
            backtrack=True
        )
        
        # Convert to sample indices
        onset_samples = (onset_frames * sr).astype(int)
        return onset_samples
    
    def slice_audio_at_onsets(self, y: np.ndarray, sr: int, onsets: np.ndarray) -> List[Tuple[int, int]]:
        """Slice audio at detected onsets."""
        slices = []
        
        # Add start and end points
        points = np.concatenate(([0], onsets, [len(y)]))
        
        # Create slices
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            duration = (end - start) / sr
            
            # Only include slices within our duration constraints
            if MIN_SAMPLE_LENGTH <= duration <= MAX_SAMPLE_LENGTH:
                slices.append((start, end))
        
        return slices
    
    def analyze_sample(self, y: np.ndarray, sr: int) -> Dict:
        """Analyze a sample and return its characteristics."""
        # Calculate features
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-6:  # Avoid division by zero
            return None
            
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_flatness = float(np.median(librosa.feature.spectral_flatness(y=y)[0]))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)))
        
        # Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = np.mean(onset_env)
        
        # Estimate pitch safely (avoid argmax shape mismatch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        if pitches.size == 0 or magnitudes.size == 0:
            pitch = 0
        else:
            flat_idx = np.argmax(magnitudes)
            freq_idx, time_idx = np.unravel_index(flat_idx, magnitudes.shape)
            pitch = pitches[freq_idx, time_idx]

        # Harmonic vs percussive energy to help spot voiced (speech/shouts)
        try:
            harm, perc = librosa.effects.hpss(y)
            harm_rms = np.sqrt(np.mean(harm**2)) if harm.size else 0
            perc_rms = np.sqrt(np.mean(perc**2)) if perc.size else 0
            harmonic_ratio = harm_rms / (harm_rms + perc_rms + 1e-9)
            percussive_ratio = perc_rms / (harm_rms + perc_rms + 1e-9)
        except Exception:
            harmonic_ratio = 0.0
            percussive_ratio = 0.0
        
        # Consonant detection: high-frequency energy ratio for sibilants (sss, sh)
        # and transient detection for plosives (t, d, k, p)
        try:
            # Sibilant detection: high freq energy (4kHz+) vs total
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            high_freq_mask = freqs > 4000
            high_freq_energy = np.mean(S[high_freq_mask, :]) if np.any(high_freq_mask) else 0
            total_energy = np.mean(S) + 1e-9
            sibilant_ratio = high_freq_energy / total_energy
            
            # Plosive detection: sharp onset at start (first 50ms has high energy spike)
            onset_samples = int(0.05 * sr)
            if len(y) > onset_samples * 2:
                onset_energy = np.sqrt(np.mean(y[:onset_samples]**2))
                body_energy = np.sqrt(np.mean(y[onset_samples:]**2)) + 1e-9
                plosive_ratio = onset_energy / body_energy
            else:
                plosive_ratio = 0.0
            
            # Consonant score: higher = more consonant-like
            consonant_score = (sibilant_ratio * 2.0) + (max(0, plosive_ratio - 1.0) * 0.5)
        except Exception:
            consonant_score = 0.0
            sibilant_ratio = 0.0
            plosive_ratio = 0.0
        
        # Pitch stability for sustained vowel detection (flat tone like "aaahh")
        try:
            # Track pitch over time and measure variance
            pitches_over_time = []
            hop_length = 512
            for i in range(0, len(y) - 2048, hop_length):
                frame = y[i:i+2048]
                p, m = librosa.piptrack(y=frame, sr=sr, n_fft=2048)
                if m.size > 0:
                    idx = np.argmax(m)
                    fi, ti = np.unravel_index(idx, m.shape)
                    frame_pitch = p[fi, ti]
                    if frame_pitch > 80:  # Valid pitch
                        pitches_over_time.append(frame_pitch)
            
            if len(pitches_over_time) >= 3:
                pitch_std = np.std(pitches_over_time)
                pitch_mean = np.mean(pitches_over_time)
                # Pitch stability: lower std = more stable/sustained
                pitch_stability = 1.0 / (1.0 + pitch_std / (pitch_mean + 1e-9))
            else:
                pitch_stability = 0.0
        except Exception:
            pitch_stability = 0.0

        # Harmonic stack metrics from spectrogram / f0
        harmonic_metrics = self.compute_harmonic_metrics(y, sr)
        
        return {
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_flatness': spectral_flatness,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zcr,
            'onset_strength': onset_strength,
            'pitch': pitch if pitch > 0 else None,
            'duration': len(y) / sr,
            'harmonic_ratio': harmonic_ratio,
            'percussive_ratio': percussive_ratio,
            'consonant_score': consonant_score,
            'sibilant_ratio': sibilant_ratio,
            'pitch_stability': pitch_stability,
            'f0_mean': harmonic_metrics.get('f0_mean'),
            'f0_std_cents': harmonic_metrics.get('f0_std_cents'),
            'voiced_prob_mean': harmonic_metrics.get('voiced_prob_mean'),
            'harmonic_stack_ratio': harmonic_metrics.get('harmonic_stack_ratio'),
            'harmonic_snr_db': harmonic_metrics.get('harmonic_snr_db'),
        }
    
    def compute_harmonic_metrics(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute harmonic-stack strength and stability from spectrogram + f0."""
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                sr=sr,
                fmin=80,
                fmax=800,
                frame_length=2048,
                hop_length=512,
            )
            if f0 is None or voiced_flag is None:
                raise ValueError("pyin returned None")
            voiced_mask = ~np.isnan(f0) & voiced_flag
            if not np.any(voiced_mask):
                raise ValueError("no voiced frames")
            f0_vals = f0[voiced_mask]
            f0_mean = float(np.mean(f0_vals))
            # std in cents relative to mean
            f0_std_cents = float(
                1200.0 * np.std(np.log2(f0_vals / (f0_mean + 1e-9)))
            )
            voiced_prob_mean = float(np.mean(voiced_prob[voiced_mask]))
        except Exception:
            return {
                'f0_mean': None,
                'f0_std_cents': None,
                'voiced_prob_mean': 0.0,
                'harmonic_stack_ratio': 0.0,
                'harmonic_snr_db': 0.0,
            }
        
        # STFT for harmonic energy
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        total_energy = float(np.sum(S) + 1e-9)
        
        harmonic_energy = 0.0
        snr_list = []
        max_harmonics = 8
        for k in range(1, max_harmonics + 1):
            target = f0_mean * k
            if target >= freqs[-1]:
                break
            bw = max(30.0, target * 0.03)  # 3% band or 30 Hz
            mask = np.abs(freqs - target) <= bw
            if not np.any(mask):
                continue
            h_energy = float(np.sum(S[mask, :]))
            harmonic_energy += h_energy
            
            # Local noise floor just outside the band
            near_mask = (np.abs(freqs - target) <= bw * 2) & (~mask)
            if np.any(near_mask):
                noise_floor = float(np.median(S[near_mask, :]) + 1e-9)
            else:
                noise_floor = 1e-9
            peak_energy = float(np.max(S[mask, :]) + 1e-9)
            snr_db = 20.0 * math.log10(peak_energy / noise_floor)
            snr_list.append(snr_db)
        
        harmonic_stack_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0
        harmonic_snr_db = float(np.mean(snr_list)) if snr_list else 0.0
        
        return {
            'f0_mean': f0_mean,
            'f0_std_cents': f0_std_cents,
            'voiced_prob_mean': voiced_prob_mean,
            'harmonic_stack_ratio': harmonic_stack_ratio,
            'harmonic_snr_db': harmonic_snr_db,
        }
    
    def categorize_sample(self, analysis: Dict) -> str:
        """Categorize a sample based on its analysis."""
        if analysis is None:
            return 'unknown'
            
        features = {
            'energy': analysis['rms'],
            'pitch': analysis['pitch'] or 0,
            'centroid': analysis['spectral_centroid'],
            'duration': analysis['duration'],
            'zcr': analysis['zero_crossing_rate'],
            'harm_ratio': analysis.get('harmonic_ratio', 0.0),
            'perc_ratio': analysis.get('percussive_ratio', 0.0),
            'consonant_score': analysis.get('consonant_score', 0.0),
            'sibilant_ratio': analysis.get('sibilant_ratio', 0.0),
            'onset_strength': analysis.get('onset_strength', 0.0),
            'pitch_stability': analysis.get('pitch_stability', 0.0),
        }
        # Dynamic vocal loudness floor
        min_rms = getattr(self, "min_rms_threshold", 0.02)
        
        # Pure vocal check: must be vowel-like (low consonant score)
        is_pure_vowel = features['consonant_score'] < 0.25
        
        vocal_like = (
            0.18 <= features['duration'] <= 1.2
            and 350 <= features['centroid'] <= 3300
            and features['harm_ratio'] > 0.4
            and features['energy'] >= min_rms * 1.3
            and features['pitch'] >= 140
            and features['pitch_stability'] >= 0.25
            and is_pure_vowel
        )
        
        # PERCUSSION DETECTION - prioritize transients
        # Kick: low frequency, strong onset, thud-like (explosions, stomps, door slams)
        # - Low spectral centroid (<1500), high energy, strong transient
        is_kick_like = (
            features['centroid'] < 1500
            and features['energy'] > 0.1
            and features['harm_ratio'] < 0.6  # More percussive than harmonic
            and features['perc_ratio'] >= 0.45
            and features['onset_strength'] >= 0.12
        )
        
        # Hihat: high frequency sibilants ("sss" sounds)
        # - High spectral centroid (>3000), high sibilant ratio
        is_hihat_like = (
            (features['sibilant_ratio'] > 0.25 or features['centroid'] > 4800 or features['zcr'] > 0.2)
            and features['duration'] <= 0.22
            and features['onset_strength'] >= 0.12
            and features['perc_ratio'] >= 0.25
        )
        
        # Snare: mid-range with sharp attack (can be explosions sped up)
        # - Mid spectral centroid, strong onset, percussive
        is_snare_like = (
            1200 < features['centroid'] < 4500
            and features['harm_ratio'] < 0.55
            and features['energy'] > 0.08
            and features['perc_ratio'] >= 0.35
            and features['onset_strength'] >= 0.1
        )
        
        # Categorization priority
        if vocal_like:
            return 'vocal'
        if is_kick_like and features['centroid'] < 1300:
            return 'kick'
        elif is_snare_like:
            return 'snare'
        elif is_hihat_like:
            return 'hihat'
        elif features['pitch'] < 200 and features['duration'] > 0.5:
            return 'bass'
        elif 200 < features['pitch'] < 1000 and features['duration'] > 0.5 and features['harm_ratio'] > 0.35:
            return 'chord'
        else:
            # Default: if harmonic, it's vocal-ish; otherwise assign to percussion
            if features['harm_ratio'] > 0.4:
                return 'vocal'
            elif features['centroid'] < 1500:
                return 'kick'  # Low = kick-ish
            elif features['centroid'] > 3500:
                return 'hihat'  # High = hihat-ish
            else:
                return 'snare'  # Mid = snare-ish
    
    def process_kick_sample(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Process a sample to be kick-like.
        - Long samples (explosions): cut shorter + fade out
        - Short samples (thuds, stomps): use as-is
        """
        duration = len(y) / sr
        
        # Target kick duration: ~0.15-0.25s
        target_duration = 0.2
        target_samples = int(target_duration * sr)
        
        if duration > 0.4:
            # Long sample (explosion-like): cut and fade
            y = y[:target_samples]
            # Apply quick fade out (last 30%)
            fade_len = int(len(y) * 0.3)
            fade_out = np.linspace(1, 0, fade_len)
            y[-fade_len:] *= fade_out
        elif duration > 0.25:
            # Medium sample: just fade the tail
            fade_len = int(len(y) * 0.4)
            fade_out = np.linspace(1, 0, fade_len)
            y[-fade_len:] *= fade_out
        # Short samples: use as-is
        
        return y
    
    def process_snare_sample(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Process a sample to be snare-like.
        - Speed up (higher pitch) for snappier sound
        - Fade out for cleaner tail
        """
        duration = len(y) / sr
        
        # Speed up by 1.3x-1.5x for snappier attack
        if duration > 0.3:
            # Speed up while preserving pitch for cleaner sound
            speed_factor = 1.4
            y = self.time_stretch_preserve_pitch(y, sr, speed_factor)
        
        # Target snare duration: ~0.15-0.2s
        target_samples = int(0.18 * sr)
        if len(y) > target_samples:
            y = y[:target_samples]
        
        # Quick fade out
        fade_len = int(len(y) * 0.25)
        if fade_len > 0:
            fade_out = np.linspace(1, 0, fade_len)
            y[-fade_len:] *= fade_out
        
        return y
    
    def process_hihat_sample(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Process a sample to be hihat-like.
        - Use sibilant portion ("sss" sounds)
        - Quick fade out so it doesn't sustain
        """
        # Target hihat duration: ~0.08-0.12s (very short)
        target_duration = 0.1
        target_samples = int(target_duration * sr)
        
        if len(y) > target_samples:
            y = y[:target_samples]
        
        # Very quick fade out (last 40%)
        fade_len = int(len(y) * 0.4)
        if fade_len > 0:
            fade_out = np.linspace(1, 0, fade_len)
            y[-fade_len:] *= fade_out
        
        return y
    
    def get_best_percussion_samples(self, samples: Dict[str, list]) -> Dict[str, Optional[Dict]]:
        """
        Get the best percussion samples with transient/onset priority.
        Always returns something for each type; adds light randomness among top candidates.
        """
        import random
        result = {'kick': None, 'snare': None, 'hihat': None}

        def score_kick(s):
            a = s.get('analysis', {})
            energy = a.get('rms', 0)
            centroid = a.get('spectral_centroid', 9999)
            onset = a.get('onset_strength', 0)
            # Low centroid good, strong onset good, loud good
            centroid_score = max(0, (1500 - centroid) / 1500)
            return energy * 3.0 + onset * 2.0 + centroid_score * 2.0

        def score_snare(s):
            a = s.get('analysis', {})
            energy = a.get('rms', 0)
            centroid = a.get('spectral_centroid', 0)
            onset = a.get('onset_strength', 0)
            # Target centroid around 2000-4000
            mid_score = max(0, 1 - abs(2500 - centroid) / 2500)
            return energy * 2.5 + onset * 2.5 + mid_score * 2.0

        def score_hihat(s):
            a = s.get('analysis', {})
            energy = a.get('rms', 0)
            centroid = a.get('spectral_centroid', 0)
            onset = a.get('onset_strength', 0)
            sibilant = a.get('sibilant_ratio', 0)
            high_score = max(0, (centroid - 3000) / 3000)
            return energy * 1.5 + onset * 2.0 + high_score * 3.0 + sibilant * 3.0

        # Helper to pick from top-N with randomness
        def pick_best(cands, scorer, top_pool=3):
            if not cands:
                return None
            scored = [(scorer(s), s) for s in cands]
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [s for _, s in scored[:max(1, top_pool)]]
            if len(pool) > 1:
                return random.choice(pool)
            return pool[0]

        result['kick'] = pick_best(samples.get('kick', []), score_kick)
        result['snare'] = pick_best(samples.get('snare', []), score_snare)
        result['hihat'] = pick_best(samples.get('hihat', []), score_hihat)

        # FALLBACK: If missing percussion, steal from other categories
        all_samples = []
        for cat in ['vocal', 'chord', 'bass']:
            all_samples.extend(samples.get(cat, []))

        if not result['kick'] and all_samples:
            # Pick lowest centroid as kick-like
            result['kick'] = min(all_samples, key=lambda s: s.get('analysis', {}).get('spectral_centroid', 9999))
            print(f"  Fallback kick from: {Path(result['kick']['path']).name}")

        if not result['hihat'] and all_samples:
            # Pick highest centroid as hihat-like
            result['hihat'] = max(all_samples, key=lambda s: s.get('analysis', {}).get('spectral_centroid', 0))
            print(f"  Fallback hihat from: {Path(result['hihat']['path']).name}")

        if not result['snare'] and all_samples:
            # Pick mid centroid as snare-like
            result['snare'] = min(all_samples, key=lambda s: abs(s.get('analysis', {}).get('spectral_centroid', 0) - 2000))
            print(f"  Fallback snare from: {Path(result['snare']['path']).name}")

        return result
    
    def save_sample(self, y: np.ndarray, category: str, index: int) -> str:
        """Save a sample to disk with proper naming."""
        category_dir = self.sample_dir / category
        category_dir.mkdir(exist_ok=True)
        
        output_path = category_dir / f"{category}_{index:03d}.wav"
        sf.write(str(output_path), y, SAMPLE_RATE)
        return str(output_path)
    
    def process_audio_file(self, audio_path: str) -> Dict[str, list]:
        """Process an audio file and extract samples."""
        # Load the audio file
        y, sr = self.load_audio(audio_path)
        # Set a dynamic RMS floor to filter weak slices
        global_rms = np.sqrt(np.mean(y**2))
        self.min_rms_threshold = max(0.015, global_rms * 0.6)
        
        # Detect onsets
        onsets = self.detect_onsets(y, sr)
        
        # Slice at onsets
        slices = self.slice_audio_at_onsets(y, sr, onsets)
        
        # Process each slice
        samples = {'kick': [], 'snare': [], 'hihat': [], 'bass': [], 'chord': [], 'vocal': []}
        
        for i, (start, end) in enumerate(slices):
            sample = y[start:end]
            # Trim leading/trailing silence (up to 100ms) for punchier entries
            max_trim = int(0.1 * sr)
            trimmed, idx = librosa.effects.trim(sample, top_db=35)
            # Cap trim to max_trim
            if idx[0] > max_trim:
                trimmed = sample[max_trim:]
            if idx[1] < len(sample) - max_trim:
                trimmed = trimmed[:len(trimmed) - max_trim]
            sample = trimmed

            # Re-check length
            if len(sample) < int(MIN_SAMPLE_LENGTH * sr):
                continue

            # Apply ultra-short fade to avoid clicks but keep attack (1 ms)
            sample = self.apply_fade(sample, fade_duration=0.001, sr=sr)
            
            # Analyze the sample
            analysis = self.analyze_sample(sample, sr)
            if analysis is None:
                continue
                
            # Categorize the sample
            category = self.categorize_sample(analysis)
            
            # Save the sample
            sample_path = self.save_sample(sample, category, i)
            
            # Store sample info
            samples[category].append({
                'path': sample_path,
                'analysis': analysis,
                'start': start / sr,
                'end': end / sr
            })
        
        return samples

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to -1 to 1 range."""
        return librosa.util.normalize(y)
    
    def apply_fade(self, y: np.ndarray, fade_duration: float = 0.01, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Apply fade in/out to avoid clicks."""
        fade_samples = int(fade_duration * sr)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        if len(y) > 2 * fade_samples:
            y[:fade_samples] *= fade_in
            y[-fade_samples:] *= fade_out
        return y

    def get_pure_vowel_samples(self, samples: Dict[str, list], top_n: int = 3,
                               window: Optional[Tuple[float, float]] = None,
                               random_pick: bool = True,
                               use_permissive: bool = True) -> List[Dict]:
        """
        Get the best pure vowel samples for chorus using smart speech detection.
        Prioritizes actual speech with clear vowel sounds, filtering out sibilants/fricatives.
        
        Args:
            samples: Dict of categorized samples
            top_n: Number of samples to return
            window: Optional (start, end) time range in seconds to filter samples
            use_permissive: Use permissive detection settings (default True for better detection)
            
        Returns:
            List of sample dicts sorted by speech quality score
        """
        # Use PERMISSIVE config by default for better speech detection
        config = SpeechDetectorConfig.permissive() if use_permissive else SpeechDetectorConfig()
        speech_detector = SpeechDetector(sr=SAMPLE_RATE, config=config)
        candidates = []
        
        print(f"\n  [SmartVocal] Analyzing {len(samples.get('vocal', []))} samples for speech quality...")
        print(f"    Using {'PERMISSIVE' if use_permissive else 'STANDARD'} detection mode")
        
        for sample in samples.get('vocal', []):
            analysis = sample.get('analysis', {})
            if analysis is None:
                continue
            
            # Require reasonable duration and some pitch stability for vowel purity
            duration = analysis.get('duration', 0)
            pitch_stability = analysis.get('pitch_stability', 0)
            if duration < 0.25 or duration > 1.5:
                continue
            if pitch_stability < 0.2:
                continue
            
            # If a window is provided, keep only samples inside it
            if window and len(window) == 2:
                w_start, w_end = window
                if not (w_start <= sample.get('start', 0) <= w_end):
                    continue
            
            # Load the sample audio for smart analysis
            sample_path = sample.get('path')
            if not sample_path or not os.path.exists(sample_path):
                continue
            
            try:
                y, sr = librosa.load(sample_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  [SmartVocal] Failed to load {sample_path}: {e}")
                continue
            
            # Use SpeechDetector to analyze phoneme content
            is_good, quality_score, phoneme = speech_detector.is_good_speech_sample(y)

            # Hard gates against harsh sibilants/consonants
            if phoneme.get('sibilant', 0) > 0.35:
                continue
            if analysis.get('sibilant_ratio', 0) > 0.35:
                continue
            if analysis.get('consonant_score', 0) > 0.35:
                continue
            
            # Additional scoring based on existing analysis
            purity_score = quality_score
            
            # Boost for harmonic content (actual speech is harmonic)
            harm_ratio = analysis.get('harmonic_ratio', 0)
            purity_score += harm_ratio * 15.0
            
            # Penalty for consonant-heavy slices
            consonant_score = analysis.get('consonant_score', 0)
            purity_score -= consonant_score * 20.0
            
            # Loudness bonus (we want the loud, clear parts)
            rms = analysis.get('rms', 0)
            purity_score += min(20, rms * 100)
            
            # Duration bonus for sustained vowels (but not too long)
            duration = analysis.get('duration', 0)
            if 0.05 <= duration <= 0.5:  # Lowered min duration
                purity_score += 10.0
            elif 0.5 < duration <= 1.0:
                purity_score += 5.0
            elif 1.0 < duration <= 1.5:
                purity_score += 2.0  # small bonus for longer sentences usable as quotes
            
            sample_with_score = sample.copy()
            sample_with_score['vowel_purity'] = max(0, purity_score)
            sample_with_score['phoneme_analysis'] = phoneme
            sample_with_score['is_good_speech'] = is_good
            candidates.append(sample_with_score)
            
            # Debug output for candidates with any quality
            if purity_score > 15:
                print(f"    {Path(sample_path).name}: quality={purity_score:.1f}, "
                      f"vowel={phoneme.get('vowel', 0):.2f}, sib={phoneme.get('sibilant', 0):.2f}, "
                      f"good={is_good}")
        
        # Sort by purity descending, prioritizing "good speech" samples
        candidates.sort(key=lambda x: (x['is_good_speech'], x['vowel_purity']), reverse=True)
        
        # Select top candidates with optional randomness for variety
        selected = []
        if candidates:
            # Prefer good speech samples
            good_speech = [c for c in candidates if c['is_good_speech']]
            if len(good_speech) >= top_n:
                top_pool = good_speech[:max(top_n * 2, top_n)]
            else:
                # Fall back to all candidates if not enough good speech
                top_pool = candidates[:max(top_n * 2, top_n)]
            
            import random
            if random_pick and len(top_pool) > top_n:
                selected = random.sample(top_pool, k=min(top_n, len(top_pool)))
                selected.sort(key=lambda x: x['vowel_purity'], reverse=True)
            else:
                selected = top_pool[:top_n]
            
            print(f"\n  [SmartVocal] Selected {len(selected)} pure vowel samples for chorus:")
            for i, c in enumerate(selected):
                ph = c.get('phoneme_analysis', {})
                print(f"    {i+1}. {Path(c['path']).name}: purity={c['vowel_purity']:.1f}, "
                      f"vowel={ph.get('vowel', 0):.2f}, sibilant={ph.get('sibilant', 0):.2f}")
        
        return selected

    def extract_smart_vocal_hits(self, audio_path: str, num_hits: int = 2,
                                   min_separation_ms: float = 100,
                                   use_permissive: bool = True) -> List[Dict]:
        """
        Extract the best vocal hits directly from audio using smart speech detection.
        This analyzes the raw audio to find transient peaks that are vowel-heavy and clear.
        
        Args:
            audio_path: Path to audio file (WAV)
            num_hits: Number of vocal hits to extract
            min_separation_ms: Minimum time between hits in milliseconds
            use_permissive: Use permissive detection settings (default True)
            
        Returns:
            List of dicts with 'audio', 'start', 'end', 'quality', 'path' info
        """
        print(f"\n  [SmartVocal] Extracting {num_hits} smart vocal hits from audio...")
        print(f"    Using {'PERMISSIVE' if use_permissive else 'STANDARD'} detection mode")
        
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"  [SmartVocal] Failed to load audio: {e}")
            return []
        
        # Use PERMISSIVE config by default for better speech detection
        config = SpeechDetectorConfig.permissive() if use_permissive else SpeechDetectorConfig()
        speech_detector = SpeechDetector(sr=SAMPLE_RATE, config=config)
        hits = speech_detector.extract_best_vocal_hits(y, num_hits=num_hits, 
                                                        min_separation_ms=min_separation_ms)
        
        if not hits:
            print("  [SmartVocal] No good vocal hits found")
            return []
        
        # Save hits as temporary WAV files and add paths
        results = []
        for i, hit in enumerate(hits):
            hit_path = str(self.sample_dir / f"smart_vocal_{i}.wav")
            sf.write(hit_path, hit['audio'], SAMPLE_RATE)
            
            hit_dict = {
                'audio': hit['audio'],
                'path': hit_path,
                'start': hit['start'],
                'end': hit['end'],
                'quality': hit['quality'],
                'phoneme': hit.get('phoneme', {}),
                'analysis': {
                    'rms': np.sqrt(np.mean(hit['audio']**2)),
                    'duration': hit['end'] - hit['start'],
                    'harmonic_ratio': hit.get('phoneme', {}).get('voiced', 0.5),
                    'consonant_score': hit.get('phoneme', {}).get('sibilant', 0) + 
                                      hit.get('phoneme', {}).get('fricative', 0),
                    'sibilant_ratio': hit.get('phoneme', {}).get('sibilant', 0),
                }
            }
            results.append(hit_dict)
            
            print(f"    Hit {i+1}: {hit['start']:.2f}s-{hit['end']:.2f}s, quality={hit['quality']:.1f}")
        
        return results

    def analyze_transient_quality(self, sample_path: str) -> Tuple[float, int]:
        """
        Analyze the transient quality of a sample.
        Returns tuple of (score, trim_samples) where:
        - score: 0-1 where 1.0 = sharp attack, 0.0 = slow attack
        - trim_samples: number of samples to trim from start to get to the main transient
        
        Measures how quickly the sample reaches peak amplitude.
        """
        try:
            y, sr = librosa.load(sample_path, sr=self.sample_rate, mono=True)
            if len(y) < 100:
                return 0.5, 0  # Too short to analyze
            
            # Get envelope using RMS
            hop_length = 64  # Fine resolution for transient detection
            frame_length = 256
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            if len(rms) < 5:
                return 0.5, 0
            
            # Find peak RMS in first 30% of sample (main transient should be early)
            search_end = max(5, len(rms) // 3)
            peak_idx = np.argmax(rms[:search_end])
            peak_val = rms[peak_idx]
            
            if peak_val < 0.01:
                return 0.5, 0  # Too quiet
            
            # Find where the signal first rises significantly (10% of peak)
            # This is the "real" start of the sound
            rise_threshold = peak_val * 0.1
            rise_frame = 0
            for i, val in enumerate(rms[:peak_idx+1]):
                if val >= rise_threshold:
                    rise_frame = i
                    break
            
            # Calculate attack time: frames from rise to 80% of peak
            threshold_80 = peak_val * 0.8
            attack_frames = 0
            for i, val in enumerate(rms[rise_frame:]):
                if val >= threshold_80:
                    attack_frames = i
                    break
            
            # Calculate trim point: samples before the rise point (the "dead air" or slow ramp)
            trim_samples = rise_frame * hop_length
            
            # Convert attack time to ms
            attack_time_ms = (attack_frames * hop_length / sr) * 1000
            
            # Score based on attack time - STRICT thresholds
            # Only very sharp attacks get high scores
            if attack_time_ms < 3:
                score = 1.0  # Excellent - instant attack
            elif attack_time_ms < 8:
                score = 0.9 - (attack_time_ms - 3) * 0.02  # 0.8-0.9
            elif attack_time_ms < 15:
                score = 0.7 - (attack_time_ms - 8) * 0.03  # 0.5-0.7
            elif attack_time_ms < 25:
                score = 0.4 - (attack_time_ms - 15) * 0.02  # 0.2-0.4
            elif attack_time_ms < 40:
                score = 0.2 - (attack_time_ms - 25) * 0.01  # 0.05-0.2
            else:
                score = max(0.0, 0.05 - (attack_time_ms - 40) * 0.002)  # Near 0
            
            return score, trim_samples
            
        except Exception as e:
            print(f"  [Transient] Error analyzing {Path(sample_path).name}: {e}")
            return 0.5, 0

    def trim_sample_attack(self, sample_path: str, output_path: str, trim_samples: int) -> Optional[str]:
        """
        Trim the beginning of a sample to remove slow attack/ramp.
        
        Args:
            sample_path: Path to input sample
            output_path: Path for trimmed output
            trim_samples: Number of samples to trim from start
            
        Returns:
            Path to trimmed sample, or None on error
        """
        try:
            if trim_samples <= 0:
                return sample_path  # Nothing to trim
            
            y, sr = librosa.load(sample_path, sr=self.sample_rate, mono=True)
            
            # Don't trim more than 20% of the sample
            max_trim = len(y) // 5
            trim_samples = min(trim_samples, max_trim)
            
            if trim_samples > 0:
                y_trimmed = y[trim_samples:]
                sf.write(output_path, y_trimmed, sr)
                trim_ms = (trim_samples / sr) * 1000
                print(f"  [Transient] Trimmed {trim_ms:.1f}ms from start of sample")
                return output_path
            
            return sample_path
            
        except Exception as e:
            print(f"  [Transient] Error trimming sample: {e}")
            return sample_path  # Return original on error

    def get_pitch_sample_candidates(self, samples: Dict[str, list], exclude_paths: Optional[List[str]] = None, 
                                    min_db: float = -25.0, top_n: int = 5) -> List[Dict]:
        """
        Get candidate samples for pitch/melody/chords selection.
        Args:
            samples: Dict of categorized samples
            exclude_paths: Paths to exclude (e.g., samples used for chorus)
            min_db: Minimum loudness threshold in dB
            top_n: Number of top candidates to return
            
        Returns:
            List of sample dicts with added 'pitch_confidence' field, sorted by confidence
        """
        candidates = []
        exclude_set = set(exclude_paths) if exclude_paths else set()
        
        # Convert min_db to linear RMS threshold
        min_rms = 10 ** (min_db / 20)
        
        # Check vocals first, then chords
        for category in ['vocal', 'chord']:
            for sample in samples.get(category, []):
                if sample.get('path') in exclude_set:
                    continue
                analysis = sample.get('analysis', {})
                if analysis is None:
                    continue
                
                # Skip quiet samples
                rms = analysis.get('rms', 0)
                if rms < min_rms:
                    continue

                # Require enough length and stability for sustained pitch
                duration = analysis.get('duration', 0)
                pitch_stability = analysis.get('pitch_stability', 0)
                f0_std_cents = analysis.get('f0_std_cents')
                voiced_prob_mean = analysis.get('voiced_prob_mean', 0.0)
                sibilant_ratio = analysis.get('sibilant_ratio', 0.0)
                perc_ratio = analysis.get('percussive_ratio', 0.0)
                spectral_flatness = analysis.get('spectral_flatness', 0.0)
                harmonic_stack = analysis.get('harmonic_stack_ratio', 0.0)
                harmonic_snr_db = analysis.get('harmonic_snr_db', 0.0)
                if duration < 0.4 or pitch_stability < 0.4:
                    continue
                if f0_std_cents is None or f0_std_cents > 35.0:
                    continue
                if voiced_prob_mean < 0.5:
                    continue
                if sibilant_ratio > 0.2:
                    continue
                if perc_ratio > 0.6:
                    continue
                if spectral_flatness > 0.25:
                    continue
                if harmonic_stack < 0.35:
                    continue
                if harmonic_snr_db < 5.0:
                    continue
                
                # Calculate "aaahh" confidence score
                # Higher = more like a sustained vowel sound
                confidence = 0.0
                
                # 1. Pitch stability (flat tone) - MOST IMPORTANT for sustained tone
                confidence += pitch_stability * 40.0  # Heavy weight
                
                # 2. Duration - prefer longer samples (sustained)
                if duration >= 0.5:
                    confidence += min(duration, 1.5) * 15.0  # Up to 22.5 points
                elif duration >= 0.3:
                    confidence += duration * 10.0
                
                # 3. Harmonic content (vowels are harmonic)
                harm_ratio = analysis.get('harmonic_ratio', 0)
                confidence += harm_ratio * 20.0
                
                # 3b. Harmonic stack strength and SNR from spectrogram
                harmonic_stack = analysis.get('harmonic_stack_ratio', 0)
                harmonic_snr_db = analysis.get('harmonic_snr_db', 0)
                confidence += harmonic_stack * 40.0  # heavy weight for clear stacks
                confidence += max(0.0, harmonic_snr_db) * 1.0  # modest boost per dB
                
                # Penalize noisy/flat spectra
                spectral_flatness = analysis.get('spectral_flatness', 0)
                spectral_rolloff = analysis.get('spectral_rolloff', 0)
                confidence -= spectral_flatness * 50.0
                if spectral_rolloff > 8000:
                    confidence -= (spectral_rolloff - 8000) / 100.0
                
                # 4. Low consonant score (no sibilants/plosives)
                consonant_score = analysis.get('consonant_score', 0)
                if consonant_score < 0.2:
                    confidence += 15.0  # Bonus for pure vowel
                elif consonant_score < 0.4:
                    confidence += 5.0
                else:
                    confidence -= 10.0  # Penalty for consonants
                
                # 5. Loudness bonus
                confidence += rms * 10.0
                
                # 6. Pitch range preference (favor mid/high speechy tone vs low rumbles)
                pitch = analysis.get('pitch')
                if pitch:
                    if prefer_mid_high and 220 <= pitch <= 600:
                        confidence += 10.0  # prefer mid/high
                    elif 150 < pitch < 800:
                        confidence += 5.0
                    # Slight penalty for very low pitches
                    if pitch < 150:
                        confidence -= 5.0
                
                # 7. Transient quality - CRITICAL for rhythmic accuracy
                # Samples with slow attacks sound off-beat - STRICT scoring
                transient_score, trim_samples = self.analyze_transient_quality(sample.get('path', ''))
                if transient_score >= 0.9:
                    confidence += 25.0  # Excellent - instant attack
                elif transient_score >= 0.7:
                    confidence += 15.0  # Good - sharp attack
                elif transient_score >= 0.5:
                    confidence += 5.0   # Okay attack
                elif transient_score >= 0.3:
                    confidence -= 10.0  # Slow attack - penalty
                else:
                    confidence -= 25.0  # Very slow attack - heavy penalty
                
                # Store transient info for later use (trimming, logging)
                sample['transient_score'] = transient_score
                sample['trim_samples'] = trim_samples
                
                # Normalize to 0-100 range roughly
                confidence = max(0, min(100, confidence))
                
                # Add confidence to sample info
                sample_with_conf = sample.copy()
                sample_with_conf['pitch_confidence'] = confidence
                candidates.append(sample_with_conf)
        
        # Sort by confidence descending
        candidates.sort(key=lambda x: x['pitch_confidence'], reverse=True)
        
        # Return top N candidates
        return candidates[:top_n]
    
    def get_best_zorammi_chord_sample(
        self,
        samples: Dict[str, list],
        exclude_paths: Optional[List[str]] = None,
        min_db: float = -20.0,
        min_duration_sec: float = 0.25,
        random_pick: bool = True,
    ) -> Tuple[Optional[Dict], bool]:
        """
        Pick a chord source for Zorammi that is long enough (>= min_duration_sec).
        Prefers longer, confident candidates; falls back to longest available.
        """
        import random
        exclude_set = set(exclude_paths) if exclude_paths else set()
        candidates = self.get_pitch_sample_candidates(samples, exclude_paths, min_db, top_n=10)

        def dur(s):
            return s.get("analysis", {}).get("duration", 0.0)

        long_enough = [c for c in candidates if dur(c) >= min_duration_sec]
        if long_enough:
            best_len = max(dur(c) for c in long_enough)
            strong = [c for c in long_enough if dur(c) >= best_len * 0.75]
            chosen_pool = strong if random_pick and len(strong) > 1 else [max(long_enough, key=dur)]
            chosen = random.choice(chosen_pool)
            print(
                f"  [ZorammiPick] candidates={len(candidates)} long_enough={len(long_enough)} "
                f"chosen={Path(chosen['path']).name} dur={dur(chosen):.3f}s"
            )
            return chosen, False

        # Fallback: pick the longest non-excluded vocal
        all_vocals = [s for s in samples.get("vocal", []) if s.get("path") not in exclude_set]
        if all_vocals:
            longest = max(all_vocals, key=dur)
            print(
                f"  [ZorammiPick][FALLBACK] No >= {min_duration_sec*1000:.0f}ms candidates. "
                f"Using longest vocal: {Path(longest['path']).name} dur={dur(longest):.3f}s"
            )
            longest_copy = longest.copy()
            longest_copy["pitch_confidence"] = 0.0
            return longest_copy, True

        return None, False
    
    def get_best_pitch_sample(self, samples: Dict[str, list], exclude_paths: Optional[List[str]] = None, 
                              min_db: float = -25.0, random_pick: bool = True,
                              min_confidence: float = 25.0) -> Tuple[Optional[Dict], bool]:
        """
        Find the best vocal/pitched sample for use as the 'pitch' instrument.
        Prioritizes sustained vowel sounds ("aaahh"): flat pitch, longer duration, harmonic content.
        
        Args:
            samples: Dict of categorized samples
            exclude_paths: Paths to exclude
            min_db: Minimum loudness threshold
            random_pick: If True, randomly pick from top candidates for variety
            min_confidence: Minimum confidence threshold to consider "good" (below = fallback)
            
        Returns:
            Tuple of (Sample info dict with 'pitch_confidence' field, is_fallback: bool)
            is_fallback=True means no good sample found, using best available anyway
        """
        import random
        
        exclude_set = set(exclude_paths) if exclude_paths else set()
        
        candidates = self.get_pitch_sample_candidates(samples, exclude_paths, min_db, top_n=5)
        
        if not candidates:
            # Ultimate fallback: try ALL vocal samples regardless of criteria
            all_vocals = [s for s in samples.get('vocal', []) if s.get('path') not in exclude_set]
            if all_vocals:
                # Just pick the loudest one that is not excluded
                best = max(all_vocals, key=lambda s: s.get('analysis', {}).get('rms', 0))
                best_copy = best.copy()
                best_copy['pitch_confidence'] = 0.0
                print(f"  [FALLBACK] No candidates found, using loudest vocal: {Path(best['path']).name}")
                return best_copy, True
            return None, False
        
        # Filter to only high-confidence candidates
        max_conf = candidates[0]['pitch_confidence']
        high_conf = [c for c in candidates if c['pitch_confidence'] >= max_conf * 0.6 and c['pitch_confidence'] >= min_confidence]
        
        # Check if we have any "good" candidates
        if not high_conf:
            # Fallback: use the best candidate anyway, but mark it as fallback
            print(f"  [FALLBACK] No high-confidence pitch samples (best={max_conf:.1f} < {min_confidence})")
            print(f"  Using best available: {Path(candidates[0]['path']).name}")
            return candidates[0], True
        
        if random_pick and len(high_conf) > 1:
            # Randomly pick from high-confidence candidates for variety
            chosen = random.choice(high_conf)
            print(f"Pitch sample candidates ({len(high_conf)} high-confidence):")
            for i, c in enumerate(high_conf):
                marker = " <-- SELECTED" if c['path'] == chosen['path'] else ""
                print(f"  {i+1}. {Path(c['path']).name}: confidence={c['pitch_confidence']:.1f}{marker}")
            return chosen, False
        else:
            print(f"Best pitch sample: {Path(candidates[0]['path']).name} (confidence={candidates[0]['pitch_confidence']:.1f})")
            return candidates[0], False
    
    def prepare_fallback_pitch_sample(self, sample_path: str, output_path: Optional[str] = None,
                                       freq: float = 293.66, stretch_rate: float = 0.2) -> Optional[str]:
        """
        Prepare a fallback pitch sample using standard pipeline:
        PitchCorrector297 -> RubberBand crisp6 -> trim 8% from start.
        Used when no good pitch sample is found.
        
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV
            freq: Target frequency for pitch correction (D4 = 293.66 Hz)
            stretch_rate: Time stretch rate (0.2 = 5x longer)
            
        Returns:
            Path to processed WAV, or None on failure
        """
        if output_path is None:
            output_path = str(self.temp_dir / "pitch_fallback.wav")
        
        return self.prepare_sample_standard(
            sample_path=sample_path,
            output_path=output_path,
            freq=freq,
            stretch_rate=stretch_rate,
            trim_start_percent=0.08,
            lowpass_freq=None,
            label="FallbackPitch"
        )

    def get_best_bass_sample(self, samples: Dict[str, list], exclude_paths: Optional[List[str]] = None,
                              random_pick: bool = True) -> Tuple[Optional[Dict], bool]:
        """
        Find the best sample for use as the bass instrument.
        Prioritizes samples with good harmonic content that can be pitched down.
        
        Args:
            samples: Dict of categorized samples
            exclude_paths: Paths to exclude
            random_pick: If True, randomly pick from top candidates
            
        Returns:
            Tuple of (Sample info dict, is_fallback: bool)
        """
        import random
        
        exclude_set = set(exclude_paths) if exclude_paths else set()
        candidates = []
        
        # Check vocals and chords for potential bass samples
        for category in ['vocal', 'chord', 'bass']:
            for sample in samples.get(category, []):
                if sample.get('path') in exclude_set:
                    continue
                analysis = sample.get('analysis', {})
                if analysis is None:
                    continue
                
                # Calculate bass suitability score
                score = 0.0

                # Hard gates to avoid percussive impacts and noisy hits
                harm_ratio = analysis.get('harmonic_ratio', 0)
                perc_ratio = analysis.get('percussive_ratio', 0)
                centroid = analysis.get('spectral_centroid', 0)
                duration = analysis.get('duration', 0)
                pitch_stability = analysis.get('pitch_stability', 0)
                sibilant_ratio = analysis.get('sibilant_ratio', 0)
                if duration < 0.3:
                    continue
                if harm_ratio < 0.25:
                    continue
                if perc_ratio > 0.55:
                    continue
                if centroid > 1800:
                    continue
                if pitch_stability < 0.25:
                    continue
                if sibilant_ratio > 0.2:
                    continue
                
                # Duration - prefer longer samples for sustained bass
                if duration >= 0.3:
                    score += min(duration, 1.0) * 20.0
                
                # Harmonic content (bass needs clean pitch)
                score += harm_ratio * 25.0
                
                # Low consonant score (clean tone)
                consonant_score = analysis.get('consonant_score', 0)
                if consonant_score < 0.3:
                    score += 15.0
                else:
                    score -= consonant_score * 10.0
                
                # Loudness
                rms = analysis.get('rms', 0)
                score += rms * 15.0
                
                # Pitch stability for sustained tones
                score += pitch_stability * 25.0

                # Penalize noisy/flat spectra
                spectral_flatness = analysis.get('spectral_flatness', 0)
                spectral_rolloff = analysis.get('spectral_rolloff', 0)
                score -= spectral_flatness * 40.0
                if spectral_rolloff > 6000:
                    score -= (spectral_rolloff - 6000) / 200.0
                
                sample_with_score = sample.copy()
                sample_with_score['bass_score'] = score
                candidates.append(sample_with_score)
        
        # Sort by score
        candidates.sort(key=lambda x: x['bass_score'], reverse=True)
        
        if not candidates:
            return None, False
        
        # Pick from top candidates
        top_pool = candidates[:min(3, len(candidates))]
        
        if random_pick and len(top_pool) > 1:
            chosen = random.choice(top_pool)
        else:
            chosen = top_pool[0]
        
        is_fallback = chosen['bass_score'] < 20.0
        
        print(f"  [Bass] Best sample: {Path(chosen['path']).name} (score={chosen['bass_score']:.1f})")
        return chosen, is_fallback

    def prepare_bass_sample(self, sample_path: str, output_path: Optional[str] = None,
                            freq: float = 73.42, stretch_rate: float = 0.2) -> Optional[str]:
        """
        Prepare a bass sample using standard pipeline:
        PitchCorrector297 -> RubberBand crisp6 -> trim 8% from start -> lowpass filter.
        
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV
            freq: Target frequency in Hz (default 73.42 = D2)
            stretch_rate: Time stretch rate (0.2 = 5x longer)
            
        Returns:
            Path to processed WAV, or None on failure
        """
        if output_path is None:
            output_path = str(self.temp_dir / "bass_corrected.wav")
        
        return self.prepare_sample_standard(
            sample_path=sample_path,
            output_path=output_path,
            freq=freq,
            stretch_rate=stretch_rate,
            trim_start_percent=0.08,
            lowpass_freq=500,  # Low-pass at 500 Hz for bass clarity
            label="Bass"
        )

    def prepare_chord_sample(self, sample_path: str, output_path: Optional[str] = None, 
                              freq: float = 293.66, stretch_rate: float = 0.15,
                              lowpass_freq: float = 2900) -> Optional[str]:
        """
        Prepare a chord/pad sample using standard pipeline with lowpass filter:
        PitchCorrector297 -> RubberBand crisp6 -> trim 8% from start -> lowpass filter.
        
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV (default: temp dir)
            freq: Target frequency in Hz (default 293.66 = D4)
            stretch_rate: Time stretch rate (0.15 = ~6.7x longer)
            lowpass_freq: Lowpass filter cutoff in Hz (2900 for gentle softening)
            
        Returns:
            Path to prepared chord sample, or None on failure.
        """
        if output_path is None:
            output_path = str(self.temp_dir / "chord_sample.wav")
        
        return self.prepare_sample_standard(
            sample_path=sample_path,
            output_path=output_path,
            freq=freq,
            stretch_rate=stretch_rate,
            trim_start_percent=0.08,
            lowpass_freq=lowpass_freq,
            label="Chord"
        )

    def prepare_zorammi_chords_sample(
        self,
        sample_path: str,
        output_path: Optional[str] = None,
        freq: float = 293.66,
        stretch_rate: float = 0.15,
    ) -> Optional[str]:
        """
        Prepare Zorammi chords sample using standard pipeline:
        PitchCorrector297 -> RubberBand crisp6 -> trim 8% from start.
        
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV (default: temp dir)
            freq: Target frequency in Hz (default 293.66 = D4)
            stretch_rate: Time stretch rate (0.15 = ~6.7x longer)
            
        Returns:
            Path to prepared Zorammi chords sample, or None on failure.
        """
        if output_path is None:
            output_path = str(self.temp_dir / "zorammi_chords.wav")
        
        return self.prepare_sample_standard(
            sample_path=sample_path,
            output_path=output_path,
            freq=freq,
            stretch_rate=stretch_rate,
            trim_start_percent=0.08,
            lowpass_freq=None,
            label="Zorammi"
        )

    def prepare_awesomeness_sample(self, sample_path: str, output_path: Optional[str] = None, 
                                     freq: float = 293.66, stretch_rate: float = 0.2) -> Optional[str]:
        """
        Prepare an awesomeness sample using standard pipeline:
        PitchCorrector297 -> RubberBand crisp6 -> trim 8% from start.
        
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV (default: temp dir)
            freq: Target frequency in Hz (default 293.66 = D4)
            stretch_rate: Time stretch rate (0.2 = 5x longer)
            
        Returns:
            Path to prepared awesomeness sample, or None on failure.
        """
        if output_path is None:
            output_path = str(self.temp_dir / "awesomeness_sample.wav")
        
        return self.prepare_sample_standard(
            sample_path=sample_path,
            output_path=output_path,
            freq=freq,
            stretch_rate=stretch_rate,
            trim_start_percent=0.08,
            lowpass_freq=None,
            label="Awesomeness"
        )

    def pitch_correct_sample(self, sample_path: str, output_path: Optional[str] = None, freq: float = 293.66) -> Optional[str]:
        """
        Run PitchCorrector297 CLI to pitch-correct a sample to D note.
        Args:
            sample_path: Path to input WAV file
            output_path: Path for output WAV (default: temp dir)
            freq: Target frequency in Hz (default 293.66 = D4)
        Returns:
            Path to pitch-corrected WAV, or None on failure.
        """
        import subprocess
        
        if output_path is None:
            output_path = str(self.temp_dir / "pitch_corrected.wav")
        
        if not PITCH_CORRECTOR_EXE.exists():
            print(f"PitchCorrector297.exe not found at {PITCH_CORRECTOR_EXE}")
            print("Please build the PitchCorrector297 project first.")
            return None
        
        try:
            result = subprocess.run(
                [str(PITCH_CORRECTOR_EXE), sample_path, output_path, str(freq)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"Pitch corrected sample saved to: {output_path}")
                return output_path
            else:
                print(f"PitchCorrector297 failed: {result.stderr or result.stdout}")
                return None
        except subprocess.TimeoutExpired:
            print("PitchCorrector297 timed out")
            return None
        except Exception as e:
            print(f"Error running PitchCorrector297: {e}")
            return None
