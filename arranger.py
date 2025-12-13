import os
import numpy as np
from pydub import AudioSegment, effects
from typing import Dict, List, Tuple, Optional, Callable
import random
from pathlib import Path
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter
from config import (
    SAMPLE_RATE, OUTPUT_DIR, BEAT_DURATION, BAR_DURATION,
    SAMPLE_CATEGORIES, SAMPLE_DIR
)

from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, CompositeVideoClip, vfx
from PIL import Image
from proglog import ProgressBarLogger

# Pillow >=10 moved ANTIALIAS; ensure compatibility for moviepy
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS


class MoviePyProgressLogger(ProgressBarLogger):
    """Custom proglog logger to capture MoviePy's video rendering progress."""
    def __init__(self, progress_fn: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._progress_fn = progress_fn
        self._last_pct = -1
    
    def bars_callback(self, bar, attr, value, old_value=None):
        """Called by proglog when progress bar updates."""
        if self._progress_fn and bar == 't' and attr == 'index':
            # Calculate percentage from current index and total
            total = self.bars.get(bar, {}).get('total', 1)
            if total > 0:
                pct = int((value / total) * 100)
                # Only update if percentage changed to avoid flooding
                if pct != self._last_pct:
                    self._last_pct = pct
                    try:
                        self._progress_fn(f"{pct}% Rendering video...")
                    except Exception as e:
                        print(f"Progress callback error: {e}")

class Arranger:
    def __init__(self, bpm: int = 140):
        self.bpm = bpm
        # Derive beat/bar duration from the requested BPM
        self.beat_duration = 60 / bpm
        self.bar_duration = self.beat_duration * 4
        self.sample_rate = SAMPLE_RATE
        self.output_dir = OUTPUT_DIR
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty audio segments for each track
        self.tracks = {
            'drums': AudioSegment.silent(duration=0),
            'bass': AudioSegment.silent(duration=0),
            'chords': AudioSegment.silent(duration=0),
            'vocals': AudioSegment.silent(duration=0),
            'pitch': AudioSegment.silent(duration=0),  # Pitch sample track
            'melody': AudioSegment.silent(duration=0),  # Melody sample track (G Minor scale)
            'bass_instrument': AudioSegment.silent(duration=0),  # Bass instrument track (D2)
            'chord_pads': AudioSegment.silent(duration=0),  # Chord pad track (chorus pads)
            'awesomeness': AudioSegment.silent(duration=0),  # Awesomeness track (last chorus only)
            'epicness_pitch': AudioSegment.silent(duration=0),  # Epicness pitch track (epicness section only)
            'zorammi_chords': AudioSegment.silent(duration=0),  # Zorammi chords track
            'fx': AudioSegment.silent(duration=0)
        }
        
        # Track the current position in the arrangement (milliseconds)
        self.current_ms = 0
        # Collect video placement events: each is dict with src/dest timing
        self.video_events = []
        
        # Pitch sample path (set by main.py after pitch correction)
        self.pitch_sample_path: Optional[str] = None
        self.pitch_sample_src_info: Optional[Dict] = None  # Original sample info for video
        
        # Melody sample path (set by main.py after pitch correction) - for top-right video
        self.melody_sample_path: Optional[str] = None
        self.melody_sample_src_info: Optional[Dict] = None  # Original sample info for video
        
        # Bass instrument sample path (set by main.py after D2 pitch correction)
        self.bass_sample_path: Optional[str] = None
        self.bass_sample_src_info: Optional[Dict] = None  # Original sample info for video
        
        # Chord pad sample path (set by main.py after time-stretch and lowpass)
        self.chord_sample_path: Optional[str] = None
        self.chord_sample_src_info: Optional[Dict] = None  # Original sample info for video

        # Zorammi chords sample path (overstretched/trimmed)
        self.zorammi_chords_sample_path: Optional[str] = None
        self.zorammi_chords_src_info: Optional[Dict] = None
        
        # Awesomeness sample path (for last chorus only - center-right position)
        self.awesomeness_sample_path: Optional[str] = None
        self.awesomeness_sample_src_info: Optional[Dict] = None
        
        # Epicness pitch sample path (for epicness section only - top-left position)
        self.epicness_pitch_sample_path: Optional[str] = None
        self.epicness_pitch_sample_src_info: Optional[Dict] = None
        
        # Percussion sample paths and info (set by main.py)
        self.kick_sample: Optional[Dict] = None
        self.snare_sample: Optional[Dict] = None
        self.hihat_sample: Optional[Dict] = None
        
        # Quote offset for video rendering (when using trimmed audio)
        self.quote_offset: float = 0.0  # Start time offset in seconds
        self.zorammi_style: bool = False  # external toggle
        
        # Instrument toggles (all enabled by default)
        self.instrument_toggles: Dict[str, bool] = {
            'kick': True, 'snare': True, 'hihat': True, 'bass': True,
            'pitch': True, 'melody': True, 'chords': True, 'vocals': True,
            'awesomeness': True, 'epicness_pitch': True, 'zorammi_chords': True
        }

    def _overlay(self, track: AudioSegment, seg: AudioSegment, position_ms: int) -> AudioSegment:
        """Overlay `seg` onto `track` at position_ms, padding if needed."""
        if len(seg) == 0:
            return track
        needed = position_ms + len(seg)
        if len(track) < needed:
            pad = AudioSegment.silent(duration=needed - len(track), frame_rate=self.sample_rate)
            track = track + pad
        return track.overlay(seg, position=position_ms)

    def position_32nd_to_ms(self, position_32nd: int) -> int:
        """Convert a position in 32nd notes to milliseconds."""
        # 1 bar = 4 beats, 1 beat = 8 32nd notes, so 1 bar = 32 32nd notes
        # At BPM, 1 beat = 60/BPM seconds, so 1 32nd note = (60/BPM)/8 seconds
        ms_per_32nd = (60 / self.bpm) / 8 * 1000
        return int(position_32nd * ms_per_32nd)

    def duration_32nd_to_ms(self, duration_32nd: int) -> int:
        """Convert a duration in 32nd notes to milliseconds."""
        ms_per_32nd = (60 / self.bpm) / 8 * 1000
        return int(duration_32nd * ms_per_32nd)
    
    def load_sample(self, file_path: str) -> AudioSegment:
        """Load a sample from disk."""
        try:
            # Load with pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                
            # Set frame rate
            audio = audio.set_frame_rate(self.sample_rate)
            
            return audio
        except Exception as e:
            print(f"Error loading sample {file_path}: {e}")
            return AudioSegment.silent(duration=100)  # Return empty audio on error
    
    def apply_highpass_filter(self, audio_segment: AudioSegment, cutoff_hz: float = 1800, 
                               order: int = 2) -> AudioSegment:
        """
        Apply a gentle high-pass filter to an audio segment.
        
        Args:
            audio_segment: Input audio
            cutoff_hz: Cutoff frequency in Hz (default 1800)
            order: Filter order (higher = steeper slope, 2 = gentle 12dB/oct)
            
        Returns:
            Filtered audio segment
        """
        try:
            # Convert AudioSegment to numpy array
            samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            
            # Normalize to -1 to 1 range
            if audio_segment.sample_width == 2:
                samples = samples / 32768.0
            elif audio_segment.sample_width == 1:
                samples = (samples - 128) / 128.0
            
            # Design Butterworth high-pass filter
            nyquist = self.sample_rate / 2
            normalized_cutoff = cutoff_hz / nyquist
            
            # Clamp to valid range
            normalized_cutoff = max(0.001, min(0.999, normalized_cutoff))
            
            # Create filter coefficients (second-order sections for stability)
            sos = butter(order, normalized_cutoff, btype='high', output='sos')
            
            # Apply filter
            filtered = sosfilt(sos, samples)
            
            # Convert back to int16
            filtered = np.clip(filtered * 32768, -32768, 32767).astype(np.int16)
            
            # Create new AudioSegment
            filtered_audio = AudioSegment(
                filtered.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            
            return filtered_audio
            
        except Exception as e:
            print(f"Error applying high-pass filter: {e}")
            return audio_segment  # Return original on error
    
    def time_to_samples(self, time_sec: float) -> int:
        """Convert time in seconds to number of samples."""
        return int(time_sec * self.sample_rate)
    
    def position_to_ms(self, position: int) -> int:
        """Convert a 16th-note position to milliseconds (no section offset)."""
        seconds_per_16th = self.beat_duration / 4  # 4 sixteenths per beat
        position_sec = position * seconds_per_16th
        return int(position_sec * 1000)
    
    def add_drum_hit(self, drum_type: str, position: int, velocity: float = 1.0, absolute_ms: Optional[int] = None) -> None:
        """
        Add a drum hit using real samples if available, otherwise synthesized.
        Records video events for percussion visuals.
        """
        position_ms = absolute_ms if absolute_ms is not None else self.position_to_ms(position)
        
        # Try to use real sample if available
        sample_info = None
        if drum_type == 'kick' and self.kick_sample:
            sample_info = self.kick_sample
        elif drum_type == 'snare' and self.snare_sample:
            sample_info = self.snare_sample
        elif drum_type == 'hihat' and self.hihat_sample:
            sample_info = self.hihat_sample
        
        if sample_info and sample_info.get('path'):
            # Use real sample
            audio_segment = self.load_sample(sample_info['path'])
            audio_segment = audio_segment + (velocity * 6 - 3)  # Velocity adjustment
            
            # Tailor processing per drum type
            dur_ms = len(audio_segment)
            if drum_type == 'kick':
                # Long booms -> cut/fade, short thuds keep as-is
                if dur_ms > 400:
                    audio_segment = audio_segment[:200].fade_out(80)
                elif dur_ms > 250:
                    audio_segment = audio_segment.fade_out(80)
            elif drum_type == 'snare':
                # Speed up and brighten; shorten (only if long enough)
                if len(audio_segment) > 200:
                    try:
                        audio_segment = audio_segment.speedup(playback_speed=1.4, crossfade=0)
                    except Exception:
                        pass  # Skip speedup if sample too short
                audio_segment = audio_segment[:180].fade_out(60)
            elif drum_type == 'hihat':
                # Apply high-pass filter around 1800Hz (gentle slope)
                audio_segment = self.apply_highpass_filter(audio_segment, cutoff_hz=1800, order=2)
                # Very short, sibilant burst
                audio_segment = audio_segment[:100].fade_out(40)
            
            # Record video event for percussion
            src_start = sample_info.get('start', 0.0)
            src_end = sample_info.get('end', src_start + 0.2)
            self.video_events.append({
                'src_start': src_start,
                'src_end': src_end,
                'dest_start_ms': position_ms,
                'dest_duration_ms': len(audio_segment),
                'type': drum_type,  # 'kick', 'snare', 'hihat'
                'sample_path': sample_info['path'],
            })
        else:
            # Fallback to synthesized
            duration_ms = 120
            t = np.linspace(0, duration_ms / 1000, int(duration_ms * self.sample_rate / 1000), False)

            if drum_type == 'kick':
                freq_start, freq_end = 100, 50
                freq = np.linspace(freq_start, freq_end, len(t))
                phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
                audio = 0.6 * np.sin(phase)
            elif drum_type == 'snare':
                noise = np.random.normal(0, 0.4, len(t))
                decay = np.linspace(1, 0, len(t))
                audio = noise * decay
            elif drum_type == 'hihat':
                noise = np.random.normal(0, 0.25, len(t))
                decay = np.linspace(1, 0, len(t))
                audio = noise * decay
            else:
                return

            envelope = np.ones_like(audio)
            attack = max(1, int(0.01 * len(envelope)))
            release = max(1, int(0.3 * len(envelope)))
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            audio = audio * envelope * velocity

            audio_segment = AudioSegment(
                (audio * 32767).astype(np.int16).tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )

        before = len(self.tracks['drums'])
        self.tracks['drums'] = self._overlay(self.tracks['drums'], audio_segment, position_ms)
        print(
            f"drum {drum_type} @pos {position} (ms->{position_ms}) "
            f"seglen={len(audio_segment)}ms before={before} after={len(self.tracks['drums'])}ms"
        )
    
    def add_bass_note(self, sample_idx: int, position: int, duration: int = 1) -> None:
        """Add a bass note to the arrangement."""
        # In a real implementation, this would load and place a bass sample
        # For now, we'll just add a placeholder
        duration_ms = self.position_to_ms(duration)  # duration is in 16th notes
        
        # Generate a simple bass note
        freq = 55 * (2 ** (sample_idx % 12 / 12))  # Walk up the chromatic scale
        t = np.linspace(0, duration_ms / 1000, int(duration_ms * self.sample_rate / 1000), False)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Apply envelope
        envelope = np.ones_like(audio)
        attack = int(0.01 * len(envelope))
        release = int(0.2 * len(envelope))
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio = audio * envelope
        
        # Convert to AudioSegment
        audio_segment = AudioSegment(
            (audio * 32767).astype(np.int16).tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Add to bass track
        position_ms = self.position_to_ms(position)
        before = len(self.tracks['bass'])
        self.tracks['bass'] = self._overlay(self.tracks['bass'], audio_segment, position_ms)
        print(
            f"bass @pos {position} (ms->{position_ms}) seglen={len(audio_segment)}ms "
            f"before={before} after={len(self.tracks['bass'])}ms"
        )
    
    def add_pitch_hit(self, position: int, semitone: int, duration: int, absolute_ms: int) -> None:
        """
        Add a pitch sample hit with semitone offset from D.
        Args:
            position: Position in 16th notes (for logging)
            semitone: Semitone offset from D (0=D, 1=D#, -2=C, etc.)
            duration: Duration in 16th notes
            absolute_ms: Absolute position in milliseconds
        """
        if not self.pitch_sample_path:
            return
        
        try:
            # Load the pitch-corrected sample using librosa for proper pitch shifting
            y, sr = librosa.load(self.pitch_sample_path, sr=self.sample_rate, mono=True)
            
            # Apply pitch shift using librosa (preserves duration)
            if semitone != 0:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone)
            
            # Calculate duration in ms
            duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)
            
            # Trim or pad to fit duration
            if len(y) > duration_samples:
                y = y[:duration_samples]
            
            # Apply fade out
            fade_samples = min(int(0.05 * sr), len(y) // 4)  # 50ms or 25% of length
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                y[-fade_samples:] *= fade
            
            # Convert numpy array to AudioSegment
            y_int16 = (y * 32767).astype(np.int16)
            audio = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # Add to pitch track
            before = len(self.tracks['pitch'])
            self.tracks['pitch'] = self._overlay(self.tracks['pitch'], audio, absolute_ms)
            print(
                f"pitch @pos {position} semitone={semitone} (ms->{absolute_ms}) seglen={len(audio)}ms "
                f"before={before} after={len(self.tracks['pitch'])}ms"
            )
            
            # Record video event if we have source info
            if self.pitch_sample_src_info:
                src_start = self.pitch_sample_src_info.get('start', 0.0)
                src_dur = self.pitch_sample_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.pitch_sample_path,
                    'type': 'pitch',
                })
        except Exception as e:
            print(f"Error adding pitch hit: {e}")
            import traceback
            traceback.print_exc()

    def add_melody_hit(self, position: int, semitone: int, duration: int, absolute_ms: int, duration_ms: int = None) -> None:
        """
        Add a melody sample hit with semitone offset from D (G Minor scale).
        Args:
            position: Position in 32nd notes (for logging)
            semitone: Semitone offset from D (G Minor: -7, -5, -4, -2, 0, 1, 3, etc.)
            duration: Duration in 32nd notes
            absolute_ms: Absolute position in milliseconds
            duration_ms: Duration in milliseconds (if None, calculated from 16th notes for backwards compat)
        """
        if not self.melody_sample_path:
            return
        
        try:
            # Load the pitch-corrected melody sample using librosa for proper pitch shifting
            y, sr = librosa.load(self.melody_sample_path, sr=self.sample_rate, mono=True)
            
            # Apply pitch shift using librosa (preserves duration)
            if semitone != 0:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone)
            
            # Use provided duration_ms or calculate from 16th notes (backwards compat)
            if duration_ms is None:
                duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)
            
            # Trim or pad to fit duration
            if len(y) > duration_samples:
                y = y[:duration_samples]
            
            # Apply fade out
            fade_samples = min(int(0.05 * sr), len(y) // 4)  # 50ms or 25% of length
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                y[-fade_samples:] *= fade
            
            # Convert numpy array to AudioSegment
            y_int16 = (y * 32767).astype(np.int16)
            audio = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # Add to melody track
            before = len(self.tracks['melody'])
            self.tracks['melody'] = self._overlay(self.tracks['melody'], audio, absolute_ms)
            print(
                f"melody @pos {position} semitone={semitone} (ms->{absolute_ms}) seglen={len(audio)}ms "
                f"before={before} after={len(self.tracks['melody'])}ms"
            )
            
            # Record video event if we have source info (top-right placement)
            if self.melody_sample_src_info:
                src_start = self.melody_sample_src_info.get('start', 0.0)
                src_dur = self.melody_sample_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.melody_sample_path,
                    'type': 'melody',
                })
        except Exception as e:
            print(f"Error adding melody hit: {e}")
            import traceback
            traceback.print_exc()

    def add_bass_instrument_hit(self, position: int, semitone: int, duration: int, absolute_ms: int) -> None:
        """
        Add a bass instrument hit with semitone offset from D2.
        Args:
            position: Position in 16th notes (for logging)
            semitone: Semitone offset from D2 (0=D, 1=D#, -2=C, 12=D3, etc.)
            duration: Duration in 16th notes
            absolute_ms: Absolute position in milliseconds
        """
        if not self.bass_sample_path:
            return
        
        try:
            # Load the D2 pitch-corrected bass sample using librosa for proper pitch shifting
            y, sr = librosa.load(self.bass_sample_path, sr=self.sample_rate, mono=True)
            
            # Apply pitch shift using librosa (preserves duration)
            if semitone != 0:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone)
            
            # Calculate duration in ms
            duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)
            
            # Trim or pad to fit duration
            if len(y) > duration_samples:
                y = y[:duration_samples]
            
            # Apply fade out
            fade_samples = min(int(0.05 * sr), len(y) // 4)  # 50ms or 25% of length
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                y[-fade_samples:] *= fade
            
            # Convert numpy array to AudioSegment
            y_int16 = (y * 32767).astype(np.int16)
            audio = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # Add to bass_instrument track
            before = len(self.tracks['bass_instrument'])
            self.tracks['bass_instrument'] = self._overlay(self.tracks['bass_instrument'], audio, absolute_ms)
            print(
                f"bass_inst @pos {position} semitone={semitone} (ms->{absolute_ms}) seglen={len(audio)}ms "
                f"before={before} after={len(self.tracks['bass_instrument'])}ms"
            )
            
            # Record video event if we have source info
            if self.bass_sample_src_info:
                src_start = self.bass_sample_src_info.get('start', 0.0)
                src_dur = self.bass_sample_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.bass_sample_path,
                    'type': 'bass_instrument',
                })
        except Exception as e:
            print(f"Error adding bass instrument hit: {e}")
            import traceback
            traceback.print_exc()

    def add_chord_hit(self, position: int, semitones: List[int], duration: int, absolute_ms: int) -> None:
        """
        Add a chord/pad hit with multiple simultaneous pitch-shifted voices.
        Args:
            position: Position in 16th notes (for logging)
            semitones: List of semitone offsets from D (each voice)
            duration: Duration in 16th notes
            absolute_ms: Absolute position in milliseconds
        """
        if not self.chord_sample_path:
            return
        
        try:
            # Load the chord sample (already time-stretched and lowpassed)
            y_base, sr = librosa.load(self.chord_sample_path, sr=self.sample_rate, mono=True)
            
            # Calculate duration in samples
            duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)
            
            # Mix all voices together
            mixed_voices = np.zeros(min(len(y_base), duration_samples), dtype=np.float32)
            
            for semitone in semitones:
                # Pitch shift each voice
                if semitone != 0:
                    y_shifted = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=semitone)
                else:
                    y_shifted = y_base.copy()
                
                # Trim to duration
                if len(y_shifted) > len(mixed_voices):
                    y_shifted = y_shifted[:len(mixed_voices)]
                elif len(y_shifted) < len(mixed_voices):
                    # Pad with zeros if sample is shorter
                    padded = np.zeros(len(mixed_voices), dtype=np.float32)
                    padded[:len(y_shifted)] = y_shifted
                    y_shifted = padded
                
                # Mix voices (average to avoid clipping)
                mixed_voices += y_shifted / len(semitones)
            
            # Apply fade out for smooth release
            fade_samples = min(int(0.1 * sr), len(mixed_voices) // 4)  # 100ms or 25% of length
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                mixed_voices[-fade_samples:] *= fade
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_voices))
            if max_val > 0:
                mixed_voices = mixed_voices / max_val * 0.8  # Leave headroom
            
            # Convert to AudioSegment
            y_int16 = (mixed_voices * 32767).astype(np.int16)
            audio = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # Add to chord_pads track
            before = len(self.tracks['chord_pads'])
            self.tracks['chord_pads'] = self._overlay(self.tracks['chord_pads'], audio, absolute_ms)
            print(
                f"chord @pos {position} voices={len(semitones)} semitones={semitones} (ms->{absolute_ms}) "
                f"seglen={len(audio)}ms before={before} after={len(self.tracks['chord_pads'])}ms"
            )
            
            # Record video event if we have source info
            if self.chord_sample_src_info:
                src_start = self.chord_sample_src_info.get('start', 0.0)
                src_dur = self.chord_sample_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.chord_sample_path,
                    'type': 'chord',
                })
        except Exception as e:
            print(f"Error adding chord hit: {e}")
            import traceback
            traceback.print_exc()

    def _design_peaking_eq(self, fc: float, q: float, gain_db: float, fs: float):
        """RBJ peaking EQ biquad, returns b, a."""
        import math
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * fc / fs
        alpha = math.sin(w0) / (2 * q)
        cos_w0 = math.cos(w0)
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])
        return b, a

    def _apply_parametric_stack(self, y: np.ndarray, sr: int, bands: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Apply a stack of peaking filters to mono signal.
        bands: list of (freq, gain_db, q)
        """
        y_out = y
        for fc, gain, q in bands:
            b, a = self._design_peaking_eq(fc, q, gain, sr)
            y_out = lfilter(b, a, y_out)
        return y_out

    def _apply_zorammi_eq_morph(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply beat-length EQ morph (start->end state) for Zorammi chords.
        We approximate by blending two statically-EQ'd versions linearly.
        """
        if len(audio) == 0:
            return audio
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.sample_width == 2:
            samples = samples / 32768.0
        elif audio.sample_width == 1:
            samples = (samples - 128) / 128.0

        sr = audio.frame_rate
        # Start and end EQ states (freq, gain_db, Q) mirroring provided curves
        start_bands = [
            (225.0, 9.0, 10.0),
            (893.0, -18.0, 8.0),
            (2507.0, 14.0, 12.0),  # boosted 3rd peak
            (3901.0, 9.0, 12.0),
            (3991.0, 6.0, 12.0),
            (8110.0, 12.0, 10.0),
            (6806.0, -10.0, 6.0),
        ]
        end_bands = [
            (680.0, 9.0, 8.0),
            (893.0, -18.0, 8.0),
            (1612.0, 14.0, 10.0),  # boosted 3rd peak
            (3152.0, 9.0, 12.0),
            (4314.0, 9.0, 12.0),
            (5160.0, 9.0, 10.0),
            (6806.0, -10.0, 6.0),
        ]

        y_start = self._apply_parametric_stack(samples, sr, start_bands)
        y_end = self._apply_parametric_stack(samples, sr, end_bands)

        # Linear morph across the segment
        ramp = np.linspace(0.0, 1.0, num=len(samples), dtype=np.float32)
        y_blend = y_start * (1.0 - ramp) + y_end * ramp

        # Normalize lightly
        max_val = np.max(np.abs(y_blend))
        if max_val > 0:
            y_blend = y_blend / max_val * 0.9

        y_int16 = np.clip(y_blend * 32767, -32768, 32767).astype(np.int16)
        return AudioSegment(
            y_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1,
        )

    def add_zorammi_chord_hit(self, position: int, semitones: List[int], duration: int, absolute_ms: int) -> None:
        """
        Add a Zorammi chord hit with parametric EQ morph per beat.
        Args:
            position: Position in 16th notes (for logging)
            semitones: List of semitone offsets from D (each voice)
            duration: Duration in 16th notes
            absolute_ms: Absolute position in milliseconds
        """
        if not self.zorammi_chords_sample_path:
            return
        try:
            y_base, sr = librosa.load(self.zorammi_chords_sample_path, sr=self.sample_rate, mono=True)

            duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)

            mixed = np.zeros(duration_samples, dtype=np.float32)
            for semitone in semitones:
                y_shifted = y_base if semitone == 0 else librosa.effects.pitch_shift(y_base, sr=sr, n_steps=semitone)
                if len(y_shifted) > duration_samples:
                    y_shifted = y_shifted[:duration_samples]
                else:
                    pad = np.zeros(duration_samples, dtype=np.float32)
                    pad[:len(y_shifted)] = y_shifted
                    y_shifted = pad
                mixed += y_shifted / max(1, len(semitones))

            # Fade out a bit
            fade_samples = min(int(0.1 * sr), len(mixed) // 4)
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                mixed[-fade_samples:] *= fade

            # To AudioSegment
            y_int16 = np.clip(mixed * 32767, -32768, 32767).astype(np.int16)
            seg = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )

            # Apply EQ morph
            seg_eq = self._apply_zorammi_eq_morph(seg)

            before = len(self.tracks['zorammi_chords'])
            self.tracks['zorammi_chords'] = self._overlay(self.tracks['zorammi_chords'], seg_eq, absolute_ms)
            print(
                f"zorammi_chord @pos {position} voices={len(semitones)} (ms->{absolute_ms}) "
                f"seglen={len(seg_eq)}ms before={before} after={len(self.tracks['zorammi_chords'])}ms"
            )

            # Record video event if available
            if self.zorammi_chords_src_info:
                src_start = self.zorammi_chords_src_info.get('start', 0.0)
                src_dur = self.zorammi_chords_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.zorammi_chords_sample_path,
                    'type': 'zorammi_chords',
                })
        except Exception as e:
            print(f"Error adding zorammi chord hit: {e}")
            import traceback
            traceback.print_exc()

    def add_awesomeness_hit(self, position: int, semitone: int, duration: int, absolute_ms: int) -> None:
        """
        Add an awesomeness sample hit with semitone offset from D.
        Used only in the last chorus, positioned center-right visually.
        Args:
            position: Position in 16th notes (for logging)
            semitone: Semitone offset from D (0=D, 1=D#, -2=C, etc.)
            duration: Duration in 16th notes
            absolute_ms: Absolute position in milliseconds
        """
        if not self.awesomeness_sample_path:
            return
        
        try:
            # Load the pitch-corrected awesomeness sample using librosa for proper pitch shifting
            y, sr = librosa.load(self.awesomeness_sample_path, sr=self.sample_rate, mono=True)
            
            # Apply pitch shift using librosa (preserves duration)
            if semitone != 0:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone)
            
            # Calculate duration in ms
            duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)
            
            # Trim or pad to fit duration
            if len(y) > duration_samples:
                y = y[:duration_samples]
            
            # Apply fade out
            fade_samples = min(int(0.05 * sr), len(y) // 4)  # 50ms or 25% of length
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                y[-fade_samples:] *= fade
            
            # Convert numpy array to AudioSegment
            y_int16 = (y * 32767).astype(np.int16)
            audio = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # Add to awesomeness track
            before = len(self.tracks['awesomeness'])
            self.tracks['awesomeness'] = self._overlay(self.tracks['awesomeness'], audio, absolute_ms)
            print(
                f"awesomeness @pos {position} semitone={semitone} (ms->{absolute_ms}) seglen={len(audio)}ms "
                f"before={before} after={len(self.tracks['awesomeness'])}ms"
            )
            
            # Record video event if we have source info (center-right placement)
            if self.awesomeness_sample_src_info:
                src_start = self.awesomeness_sample_src_info.get('start', 0.0)
                src_dur = self.awesomeness_sample_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.awesomeness_sample_path,
                    'type': 'awesomeness',
                })
        except Exception as e:
            print(f"Error adding awesomeness hit: {e}")
            import traceback
            traceback.print_exc()

    def add_epicness_pitch_hit(self, position: int, semitone: int, duration: int, absolute_ms: int) -> None:
        """
        Add an epicness pitch sample hit with semitone offset from D.
        Used only in the epicness section, positioned top-left visually.
        Args:
            position: Position in 16th notes (for logging)
            semitone: Semitone offset from D (0=D, 1=D#, -2=C, etc.)
            duration: Duration in 16th notes
            absolute_ms: Absolute position in milliseconds
        """
        if not self.epicness_pitch_sample_path:
            return
        
        try:
            # Load the pitch-corrected epicness pitch sample using librosa for proper pitch shifting
            y, sr = librosa.load(self.epicness_pitch_sample_path, sr=self.sample_rate, mono=True)
            
            # Apply pitch shift using librosa (preserves duration)
            if semitone != 0:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone)
            
            # Calculate duration in ms
            duration_ms = self.position_to_ms(duration)
            duration_samples = int(duration_ms / 1000 * sr)
            
            # Trim or pad to fit duration
            if len(y) > duration_samples:
                y = y[:duration_samples]
            
            # Apply fade out
            fade_samples = min(int(0.05 * sr), len(y) // 4)  # 50ms or 25% of length
            if fade_samples > 0:
                fade = np.linspace(1, 0, fade_samples)
                y[-fade_samples:] *= fade
            
            # Convert numpy array to AudioSegment
            y_int16 = (y * 32767).astype(np.int16)
            audio = AudioSegment(
                y_int16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            
            # Add to epicness_pitch track
            before = len(self.tracks['epicness_pitch'])
            self.tracks['epicness_pitch'] = self._overlay(self.tracks['epicness_pitch'], audio, absolute_ms)
            print(
                f"epicness_pitch @pos {position} semitone={semitone} (ms->{absolute_ms}) seglen={len(audio)}ms "
                f"before={before} after={len(self.tracks['epicness_pitch'])}ms"
            )
            
            # Record video event if we have source info (top-left placement, same as pitch but different sample)
            if self.epicness_pitch_sample_src_info:
                src_start = self.epicness_pitch_sample_src_info.get('start', 0.0)
                src_dur = self.epicness_pitch_sample_src_info.get('analysis', {}).get('duration', 0.5)
                self.video_events.append({
                    'src_start': src_start,
                    'src_end': src_start + src_dur,
                    'dest_start_ms': absolute_ms,
                    'dest_duration_ms': duration_ms,
                    'sample_path': self.epicness_pitch_sample_path,
                    'type': 'epicness_pitch',
                })
        except Exception as e:
            print(f"Error adding epicness pitch hit: {e}")
            import traceback
            traceback.print_exc()

    def add_vocal_chop(self, sample_path: str, position: int, duration: int = 1, absolute_ms: Optional[int] = None) -> None:
        """Add a vocal chop to the arrangement."""
        try:
            # Load the sample
            audio = self.load_sample(sample_path)
            
            # Calculate duration in milliseconds
            duration_ms = self.position_to_ms(duration)  # duration is in 16th notes
            
            # Trim or loop the sample to fit the duration
            if len(audio) > duration_ms:
                audio = audio[:duration_ms]
            
            # Add to vocals track
            position_ms = absolute_ms if absolute_ms is not None else self.position_to_ms(position)
            before = len(self.tracks['vocals'])
            self.tracks['vocals'] = self._overlay(self.tracks['vocals'], audio, position_ms)
            print(
                f"vocal @pos {position} (ms->{position_ms}) seglen={len(audio)}ms "
                f"before={before} after={len(self.tracks['vocals'])}ms"
            )
        except Exception as e:
            print(f"Error adding vocal chop: {e}")
    
    def render_section(self, section: Dict, samples: Dict[str, list]) -> None:
        """Render a complete section of the song."""
        section_name = section['name']
        print(f"Rendering section: {section_name}")
        
        # Start time in ms for this section
        section_start_ms = int(self.current_ms)
        
        chorus_only = section_name == 'chorus'

        # Process drums for all sections (check toggles)
        if 'drums' in section:
            for drum_type, positions in section['drums'].items():
                # Check if this drum type is enabled
                if drum_type == 'kick' and not self.instrument_toggles.get('kick', True):
                    continue
                if drum_type == 'snare' and not self.instrument_toggles.get('snare', True):
                    continue
                if drum_type == 'hihat' and not self.instrument_toggles.get('hihat', True):
                    continue
                # Skip hihat from pattern if this is chorus (we'll add 16th notes instead)
                if drum_type == 'hihat' and chorus_only:
                    continue
                for pos in positions:
                    absolute_ms = section_start_ms + self.position_to_ms(pos)
                    self.add_drum_hit(drum_type, pos, absolute_ms=absolute_ms)
        
        # Add 16th note hihats throughout chorus with velocity variation (if enabled)
        if chorus_only and self.instrument_toggles.get('hihat', True):
            num_bars = section['length_bars']
            total_16ths = num_bars * 16
            for i in range(total_16ths):
                absolute_ms = section_start_ms + self.position_to_ms(i)
                # Even notes (0, 2, 4...) are quieter (-8dB), odd notes (1, 3, 5...) are full
                # In the screenshot, it looks like alternating velocity pattern
                if i % 2 == 1:  # Odd positions (1, 3, 5...) - quieter
                    velocity = 0.4  # About -8dB quieter
                else:  # Even positions (0, 2, 4...) - full volume
                    velocity = 1.0
                self.add_drum_hit('hihat', i, velocity=velocity, absolute_ms=absolute_ms)
        
        # Process bass (skip in chorus-only mode)
        if not chorus_only and 'bass' in section and samples.get('bass'):
            for pos, sample_idx in section['bass']:
                if sample_idx < len(samples['bass']):
                    sample_path = samples['bass'][sample_idx]['path']
                    self.add_vocal_chop(sample_path, pos, duration=2)  # Bass notes are longer
        
        # Process vocals (if enabled)
        if 'vocals' in section and samples.get('vocal') and self.instrument_toggles.get('vocals', True):
            for item in section['vocals']:
                # item can be (pos, sample_idx) or (pos, sample_idx, duration)
                if len(item) == 3:
                    pos, sample_idx, dur = item
                else:
                    pos, sample_idx = item
                    dur = 1
                if sample_idx < len(samples['vocal']):
                    sample_path = samples['vocal'][sample_idx]['path']
                    absolute_ms = section_start_ms + self.position_to_ms(pos)
                    self.add_vocal_chop(sample_path, pos, duration=dur, absolute_ms=absolute_ms)
                    # Capture video event using original sample timing info
                    sample_info = samples['vocal'][sample_idx]
                    self.video_events.append({
                        'src_start': sample_info.get('start', 0.0),
                        'src_end': sample_info.get('end', sample_info.get('start', 0.0) + (dur * self.beat_duration / 4)),
                        'dest_start_ms': absolute_ms,
                        'dest_duration_ms': self.position_to_ms(dur),
                        'sample_path': sample_path,
                    })
                else:
                    print(f"Skip vocal: sample_idx {sample_idx} out of range {len(samples['vocal'])}")

        # Process pitch events (only in chorus, if enabled)
        if 'pitch' in section and section['pitch'] and self.instrument_toggles.get('pitch', True):
            for item in section['pitch']:
                # item is (position_16th, semitone_offset, duration_16ths)
                pos, semitone, dur = item
                absolute_ms = section_start_ms + self.position_to_ms(pos)
                self.add_pitch_hit(pos, semitone, dur, absolute_ms)

        # Process bass instrument events (chorus, epicness, dundundenden, if enabled)
        if 'bass_instrument' in section and section['bass_instrument'] and self.instrument_toggles.get('bass', True):
            for item in section['bass_instrument']:
                # item is (position_16th, semitone_offset, duration_16ths)
                pos, semitone, dur = item
                absolute_ms = section_start_ms + self.position_to_ms(pos)
                self.add_bass_instrument_hit(pos, semitone, dur, absolute_ms)

        # Process melody events (chorus + dundundenden, if enabled)
        if 'melody' in section and section['melody'] and self.instrument_toggles.get('melody', True):
            for item in section['melody']:
                # item is (position_32nds, semitone_offset, duration_32nds)
                pos_32nd, semitone, dur_32nd = item
                absolute_ms = section_start_ms + self.position_32nd_to_ms(pos_32nd)
                duration_ms = self.duration_32nd_to_ms(dur_32nd)
                self.add_melody_hit(pos_32nd, semitone, dur_32nd, absolute_ms, duration_ms)

        # Process Zorammi chords
        if self.zorammi_style and 'zorammi_chords' in section and section['zorammi_chords'] and self.instrument_toggles.get('zorammi_chords', True):
            for pos, semitones, dur in section['zorammi_chords']:
                absolute_ms = section_start_ms + self.position_to_ms(pos)
                self.add_zorammi_chord_hit(pos, semitones, dur, absolute_ms)

        # Process chord pad events (chorus only - multi-voice pads, if enabled)
        if 'chord_pads' in section and section['chord_pads'] and self.instrument_toggles.get('chords', True):
            for item in section['chord_pads']:
                # item is (position_16th, [semitone_offsets], duration_16ths)
                pos, semitones, dur = item
                absolute_ms = section_start_ms + self.position_to_ms(pos)
                self.add_chord_hit(pos, semitones, dur, absolute_ms)

        # Process awesomeness events (last chorus only - center-right position, if enabled)
        if 'awesomeness' in section and section['awesomeness'] and self.instrument_toggles.get('awesomeness', True):
            for item in section['awesomeness']:
                # item is (position_16th, semitone_offset, duration_16ths)
                pos, semitone, dur = item
                absolute_ms = section_start_ms + self.position_to_ms(pos)
                self.add_awesomeness_hit(pos, semitone, dur, absolute_ms)

        # Process epicness pitch events (epicness section only - top-left position, if enabled)
        if 'epicness_pitch' in section and section['epicness_pitch'] and self.instrument_toggles.get('epicness_pitch', True):
            for item in section['epicness_pitch']:
                # item is (position_16th, semitone_offset, duration_16ths)
                pos, semitone, dur = item
                absolute_ms = section_start_ms + self.position_to_ms(pos)
                self.add_epicness_pitch_hit(pos, semitone, dur, absolute_ms)

        # Debug lengths after this section
        print(
            f"Track lengths after {section_name}: "
            f"drums={len(self.tracks['drums'])}ms, "
            f"bass={len(self.tracks['bass'])}ms, "
            f"chords={len(self.tracks['chords'])}ms, "
            f"vocals={len(self.tracks['vocals'])}ms, "
            f"pitch={len(self.tracks['pitch'])}ms, "
            f"melody={len(self.tracks['melody'])}ms, "
            f"bass_inst={len(self.tracks['bass_instrument'])}ms, "
            f"chord_pads={len(self.tracks['chord_pads'])}ms, "
            f"zorammi_chords={len(self.tracks['zorammi_chords'])}ms, "
            f"awesomeness={len(self.tracks['awesomeness'])}ms, "
            f"epicness_pitch={len(self.tracks['epicness_pitch'])}ms"
        )
        
        # Update current position (ms)
        self.current_ms += section['length_bars'] * self.bar_duration * 1000
    
    def apply_effects(self, audio: AudioSegment) -> AudioSegment:
        """Apply effects to the audio."""
        # Normalize
        audio = effects.normalize(audio)
        
        # Add some compression
        audio = audio.compress_dynamic_range(threshold=-20, ratio=4.0, attack=5, release=50)
        
        # Add a bit of reverb (simulated with a simple echo)
        # In a real implementation, you'd use a proper reverb effect
        if len(audio) > 50:  # Ensure audio is long enough
            echo = audio[-50:].apply_gain(-10)
            audio = audio.overlay(echo, position=50)
        
        return audio
    
    def mixdown(self, output_path: Optional[str] = None, base_bed_path: Optional[str] = None) -> str:
        """Mix all tracks together and export the final audio."""
        if output_path is None:
            output_path = str(self.output_dir / "sparta_remix.wav")
        
        # Create a silent audio segment for the full length (ensure non-zero)
        max_length = max(len(track) for track in self.tracks.values())
        if max_length <= 0:
            # Fallback: 2 seconds of silence to avoid zero-byte exports
            max_length = 2000
        mixed = AudioSegment.silent(duration=max_length, frame_rate=self.sample_rate)

        print(
            f"Mixdown lengths (ms): drums={len(self.tracks['drums'])}, "
            f"bass={len(self.tracks['bass'])}, chords={len(self.tracks['chords'])}, "
            f"vocals={len(self.tracks['vocals'])}, pitch={len(self.tracks['pitch'])}, "
            f"melody={len(self.tracks['melody'])}, bass_inst={len(self.tracks['bass_instrument'])}, "
            f"chord_pads={len(self.tracks['chord_pads'])}, zorammi_chords={len(self.tracks['zorammi_chords'])}, "
            f"awesomeness={len(self.tracks['awesomeness'])}, "
            f"epicness_pitch={len(self.tracks['epicness_pitch'])}, fx={len(self.tracks['fx'])}, max={max_length}"
        )
        
        # Mix all tracks
        for name, track in self.tracks.items():
            if len(track) > 0:
                # Apply track-specific processing
                if name == 'drums':
                    track = track.apply_gain(-3)  # Slightly lower volume for drums
                elif name == 'bass':
                    track = track.low_pass_filter(250)  # Keep only low frequencies for bass
                elif name == 'vocals':
                    track = track.high_pass_filter(200)  # Remove low frequencies from vocals
                elif name == 'pitch':
                    # Pitch sample: slight gain boost, leave full frequency range
                    track = track.apply_gain(2)
                elif name == 'melody':
                    # Melody sample: slight gain boost for clarity
                    track = track.apply_gain(1)
                elif name == 'bass_instrument':
                    # Bass instrument: low-pass filter and slight boost for warmth
                    track = track.low_pass_filter(400)
                    track = track.apply_gain(3)
                elif name == 'chord_pads':
                    # Chord pads: soft pad sound, slightly lowered for background
                    track = track.apply_gain(-2)  # Keep it as background texture
                elif name == 'awesomeness':
                    # Awesomeness track: slight gain boost for presence
                    track = track.apply_gain(2)
                elif name == 'epicness_pitch':
                    # Epicness pitch track: slight gain boost
                    track = track.apply_gain(2)
                elif name == 'zorammi_chords':
                    # Zorammi chords: leave bright, small gain boost
                    track = track.apply_gain(2)
                
                # Apply effects
                track = self.apply_effects(track)
                
                # Add to mix
                mixed = mixed.overlay(track)
        
        # Optional base bed overlay (e.g., base.wav) at -5 dB
        if base_bed_path and os.path.exists(base_bed_path):
            try:
                bed = AudioSegment.from_file(base_bed_path).set_frame_rate(self.sample_rate).set_channels(2)
                bed = bed.apply_gain(-5.0)
                if len(bed) < max_length:
                    bed = bed.append(AudioSegment.silent(duration=max_length - len(bed), frame_rate=self.sample_rate), crossfade=0)
                mixed = mixed.overlay(bed)
                print(f"Applied base bed from {base_bed_path} at -5 dB")
            except Exception as e:
                print(f"Failed to load base bed {base_bed_path}: {e}")

        # Apply master effects
        mixed = self.apply_effects(mixed)
        
        # Export the final mix
        mixed.export(output_path, format="wav")
        print(f"Exported mix to {output_path}")
        return output_path

    def _preconvert_video(self, source_video_path: str, resolution: Tuple[int, int] = (854, 480), fps: int = 30) -> str:
        """
        Pre-convert source video to 480p 30fps for faster processing.
        Returns the path to the converted video.
        """
        import tempfile
        import subprocess
        
        # Create temp file for converted video
        temp_dir = Path(tempfile.gettempdir())
        converted_path = str(temp_dir / "sparta_preconverted.mp4")
        
        print(f"Pre-converting video to {resolution[1]}p {fps}fps...")
        
        try:
            ffmpeg_bin = os.environ.get("IMAGEIO_FFMPEG_EXE") or os.environ.get("FFMPEG_BINARY") or "ffmpeg"
            if ffmpeg_bin and os.path.exists(ffmpeg_bin):
                ffmpeg_cmd = ffmpeg_bin
            else:
                ffmpeg_cmd = "ffmpeg"

            # Use ffmpeg for fast conversion
            cmd = [
                ffmpeg_cmd, '-y',
                '-i', source_video_path,
                '-vf', f'scale={resolution[0]}:{resolution[1]}',
                '-r', str(fps),
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                converted_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg warning: {result.stderr[:500]}")
            
            if os.path.exists(converted_path):
                print(f"Pre-conversion complete: {converted_path}")
                return converted_path
            else:
                print("Pre-conversion failed, using original video")
                return source_video_path
                
        except FileNotFoundError:
            print("FFmpeg not found, using moviepy for conversion...")
            try:
                clip = VideoFileClip(source_video_path)
                clip_resized = clip.resize(resolution).set_fps(fps)
                clip_resized.write_videofile(
                    converted_path,
                    codec='libx264',
                    preset='ultrafast',
                    audio_codec='aac',
                    fps=fps
                )
                clip.close()
                clip_resized.close()
                print(f"Pre-conversion complete: {converted_path}")
                return converted_path
            except Exception as e:
                print(f"Moviepy conversion failed: {e}, using original video")
                return source_video_path
        except Exception as e:
            print(f"Pre-conversion failed: {e}, using original video")
            return source_video_path

    def render_video(
        self,
        source_video_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        resolution: Tuple[int, int] = (854, 480),
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Render a video aligned to the generated audio.
        - Black background for rests.
        - Overlay source video subclips per recorded video_events.
        - Pitch events appear in top-left corner at 1/4 size.
        - Encode H.264 at the requested resolution.
        """
        if output_path is None:
            output_path = str(self.output_dir / "sparta_remix.mp4")

        def _configure_moviepy_ffmpeg():
            ffmpeg_bin = os.environ.get("IMAGEIO_FFMPEG_EXE") or os.environ.get("FFMPEG_BINARY")
            if ffmpeg_bin and os.path.exists(ffmpeg_bin):
                try:
                    from moviepy.config import change_settings
                    change_settings({"FFMPEG_BINARY": ffmpeg_bin})
                except Exception:
                    pass

        # Create progress logger for video rendering
        video_logger = MoviePyProgressLogger(progress_callback) if progress_callback else "bar"

        def _render_black_fallback() -> Optional[str]:
            try:
                _configure_moviepy_ffmpeg()
                audio_clip = AudioFileClip(audio_path)
                total_duration = audio_clip.duration
                base_clip = ColorClip(size=resolution, color=(0, 0, 0), duration=0).set_duration(total_duration)
                composite = base_clip.set_audio(audio_clip)
                composite.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    fps=30,
                    preset="medium",
                    logger=video_logger
                )
                try:
                    composite.close()
                except Exception:
                    pass
                try:
                    audio_clip.close()
                except Exception:
                    pass
                try:
                    base_clip.close()
                except Exception:
                    pass
                print(f"Exported video to {output_path}")
                return output_path
            except Exception as e:
                print(f"Video render failed (black fallback): {e}")
                return None

        print(f"Video events recorded: {len(self.video_events)}")
        if not self.video_events:
            print("No video events recorded; rendering black video with audio.")
            return _render_black_fallback()

        try:
            _configure_moviepy_ffmpeg()
            # Pre-convert video to 480p 30fps for faster processing
            converted_video_path = self._preconvert_video(source_video_path, resolution, fps=30)
            
            base_clip = ColorClip(size=resolution, color=(0, 0, 0), duration=0)
            src = VideoFileClip(converted_video_path)
            src_duration = src.duration
            print(f"Source video duration: {src_duration:.2f}s, quote_offset: {self.quote_offset:.2f}s")

            # Calculate small sample size (1/4 of output resolution)
            small_size = (resolution[0] // 4, resolution[1] // 4)
            
            # Positions for small clips (bottom row for percussion, top corners for pitch/melody, top-center for bass)
            # Pitch: top-left (0, 0)
            # Melody: top-right (G Minor scale melodies)
            # Bass instrument: top-center (with B&W filter)
            # Chord: center-left, below pitch (second row left)
            # Awesomeness: center-right (last chorus only)
            # Epicness pitch: top-left (same as pitch, but different sample for epicness section)
            # Kick: bottom-left
            # Hihat: bottom-middle  
            # Snare: bottom-right
            pos_pitch = (0, 0)
            pos_melody = (resolution[0] - small_size[0], 0)  # Top-right
            pos_bass_inst = (resolution[0] // 2 - small_size[0] // 2, 0)  # Center-top
            pos_chord = (0, small_size[1])  # Center-left, below pitch (second row)
            pos_awesomeness = (resolution[0] - small_size[0], small_size[1])  # Center-right (second row right)
            pos_epicness_pitch = (0, 0)  # Top-left (same position as pitch)
            pos_zorammi_chords = (resolution[0] - small_size[0], 0)  # Top-right
            pos_kick = (0, resolution[1] - small_size[1])
            pos_hihat = (resolution[0] // 2 - small_size[0] // 2, resolution[1] - small_size[1])
            pos_snare = (resolution[0] - small_size[0], resolution[1] - small_size[1])

            clips = []
            # Ensure base duration matches audio
            audio_clip = AudioFileClip(audio_path)
            total_duration = audio_clip.duration
            base_clip = base_clip.set_duration(total_duration)
            clips.append(base_clip)

            # Track indices for alternating flip
            pitch_idx = 0
            melody_idx = 0
            vocal_idx = 0
            kick_idx = 0
            snare_idx = 0
            hihat_idx = 0
            bass_inst_idx = 0
            chord_idx = 0
            awesomeness_idx = 0
            epicness_pitch_idx = 0
            zorammi_chords_idx = 0
            
            for i, ev in enumerate(self.video_events):
                # Add quote offset to source times (samples are relative to trimmed audio)
                start = ev["src_start"] + self.quote_offset
                end = ev["src_end"] + self.quote_offset
                dest_start = ev["dest_start_ms"] / 1000.0
                dest_dur = ev["dest_duration_ms"] / 1000.0
                event_type = ev.get("type", "vocal")
                
                # Prevent zero/negative or out-of-bounds
                if dest_dur <= 0:
                    continue
                if start < 0 or end > src_duration or start >= end:
                    continue
                
                try:
                    if event_type == "pitch":
                        # Pitch samples: small box in top-left corner
                        sub = src.subclip(start, end).resize(small_size)
                        if pitch_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_pitch).set_start(dest_start).set_duration(dest_dur)
                        pitch_idx += 1
                    elif event_type == "melody":
                        # Melody samples: small box in top-right corner (G Minor melodies)
                        sub = src.subclip(start, end).resize(small_size)
                        if melody_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_melody).set_start(dest_start).set_duration(dest_dur)
                        melody_idx += 1
                    elif event_type == "bass_instrument":
                        # Bass instrument: small box in top-center with B&W filter
                        sub = src.subclip(start, end).resize(small_size)
                        # Apply black and white filter
                        sub = sub.fx(vfx.blackwhite)
                        if bass_inst_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_bass_inst).set_start(dest_start).set_duration(dest_dur)
                        bass_inst_idx += 1
                    elif event_type == "kick":
                        # Kick: bottom-left corner
                        sub = src.subclip(start, end).resize(small_size)
                        if kick_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_kick).set_start(dest_start).set_duration(dest_dur)
                        kick_idx += 1
                    elif event_type == "hihat":
                        # Hihat: bottom-middle
                        sub = src.subclip(start, end).resize(small_size)
                        if hihat_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_hihat).set_start(dest_start).set_duration(dest_dur)
                        hihat_idx += 1
                    elif event_type == "snare":
                        # Snare: bottom-right corner
                        sub = src.subclip(start, end).resize(small_size)
                        if snare_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_snare).set_start(dest_start).set_duration(dest_dur)
                        snare_idx += 1
                    elif event_type == "chord":
                        # Chord pads: center-left, below pitch visuals
                        sub = src.subclip(start, end).resize(small_size)
                        if chord_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_chord).set_start(dest_start).set_duration(dest_dur)
                        chord_idx += 1
                    elif event_type == "awesomeness":
                        # Awesomeness: center-right (last chorus only)
                        sub = src.subclip(start, end).resize(small_size)
                        if awesomeness_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_awesomeness).set_start(dest_start).set_duration(dest_dur)
                        awesomeness_idx += 1
                    elif event_type == "epicness_pitch":
                        # Epicness pitch: top-left (epicness section only)
                        sub = src.subclip(start, end).resize(small_size)
                        if epicness_pitch_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_epicness_pitch).set_start(dest_start).set_duration(dest_dur)
                        epicness_pitch_idx += 1
                    elif event_type == "zorammi_chords":
                        # Zorammi chords: top-right, quarter size
                        sub = src.subclip(start, end).resize(small_size)
                        if zorammi_chords_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_position(pos_zorammi_chords).set_start(dest_start).set_duration(dest_dur)
                        zorammi_chords_idx += 1
                    else:
                        # Regular vocal samples: full size
                        sub = src.subclip(start, end).resize(resolution)
                        if vocal_idx % 2 == 1:
                            sub = sub.fx(vfx.mirror_x)
                        sub = sub.set_start(dest_start).set_duration(dest_dur)
                        vocal_idx += 1
                    
                    clips.append(sub)
                except Exception as clip_err:
                    print(f"  Event {i}: error creating clip: {clip_err}")

            print(f"Total clips for composite: {len(clips)} (1 base + {len(clips)-1} events)")
            composite = CompositeVideoClip(clips, size=resolution)
            composite = composite.set_audio(audio_clip)
            composite.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=30,
                preset="medium",
                logger=video_logger
            )
            try:
                composite.close()
            except Exception:
                pass
            try:
                audio_clip.close()
            except Exception:
                pass
            try:
                src.close()
            except Exception:
                pass
            try:
                base_clip.close()
            except Exception:
                pass
            print(f"Exported video to {output_path}")
            return output_path
        except Exception as e:
            print(f"Video render failed: {e}")
            return _render_black_fallback()
