import random
from typing import List, Dict, Tuple, Optional
import numpy as np
from pydub import AudioSegment, effects
import os

from config import PATTERNS, BPM, BEAT_DURATION, BAR_DURATION, SAMPLE_RATE

class PatternGenerator:
    def __init__(self, bpm: int = BPM, variation_enabled: bool = True, variation_seed: Optional[int] = None, zorammi_style: bool = False):
        self.bpm = bpm
        self.beat_duration = 60 / bpm  # in seconds
        self.bar_duration = self.beat_duration * 4  # 4/4 time
        self.variation_enabled = variation_enabled
        self.variation_seed = variation_seed
        self.zorammi_style = zorammi_style
        
        # G Minor scale semitone offsets from D (root)
        # G, A, Bb, C, D, Eb, F = -7, -5, -4, -2, 0, 1, 3
        self.g_minor_scale = [-7, -5, -4, -2, 0, 1, 3]
        # Extended with octave up versions
        self.g_minor_extended = [-7, -5, -4, -2, 0, 1, 3, 5, 7, 8, 10, 12, 13, 15]
        
        # Vocal pattern strings per section
        self.vocal_patterns = {
            # Intro: 1*______2*______3*______1_1_1111 (2 bars = 32 sixteenths)
            "intro": ["1*______2*______3*______1_1_1111"],
            "chorus": ["11_11_111_1_1_11222_2_222_222_2_"],  # 4 bars looped
            "dundundenden": [
                # 2-bar motif; we will loop it to fill 6 bars
                "1*__2*__3*__1111" "1*__2*__3*__1_11"
            ],
            "epicness": [
                # Line A
                "1*__1*332*1_1_11__1_1*113_3_22221*3_1*332*1_1_111_1111111111111111",
                # Line B (rests then sparse hits)
                "____________________________________________3_3_____3__3_3333_33__",
            ],
        }
        
        # Pitch patterns for chorus and dundundenden
        # Format: list of semitone offsets or None for rest, each element = 8th note
        # 2 bars = 16 eighth notes per pattern
        # Multiple patterns per section - randomly selected every 2 bars
        
        # CHORUS patterns (randomly rotated every 2 bars)
        # Pattern notation: 0*__0*__1*__1*__-2*__-2*__1*__1*__
        self.chorus_pitch_patterns = [
            # 0*__0*__1*__1*__-2*__-2*__1*__1*__ (basic sparse pattern)
            [0, None, 0, None, 1, None, 1, None, -2, None, -2, None, 1, None, 1, None],
            # 0*7*0*7*1*8*1*8*-2*5*-2*5*1*8*1*8* (filled with octave harmony)
            [0, 7, 0, 7, 1, 8, 1, 8, -2, 5, -2, 5, 1, 8, 1, 8],
            # 0*12*0*12*1*13*1*13*-2*10*-2*10*1*13*1*13* (higher octave harmony)
            [0, 12, 0, 12, 1, 13, 1, 13, -2, 10, -2, 10, 1, 13, 1, 13],
        ]
        
        # DUNDUNDENDEN patterns (randomly rotated every 2 bars)
        self.dundundenden_pitch_patterns = [
            # 0*__0*__1*__1*__-2*__-2*__1*__1*__
            [0, None, 0, None, 1, None, 1, None, -2, None, -2, None, 1, None, 1, None],
            # 2*__2*__3*__3*__0*__0*__3*__3*__
            [2, None, 2, None, 3, None, 3, None, 0, None, 0, None, 3, None, 3, None],
            # 1*__1*__3*__3*__0*__0*__3*__3*__
            [1, None, 1, None, 3, None, 3, None, 0, None, 0, None, 3, None, 3, None],
            # 7*__7*__8*__8*__5*__5*__8*__8*__
            [7, None, 7, None, 8, None, 8, None, 5, None, 5, None, 8, None, 8, None],
            # 12*__12*__13*__13*__10*__10*__13*__13*__
            [12, None, 12, None, 13, None, 13, None, 10, None, 10, None, 13, None, 13, None],
        ]
        
        # BASS patterns - pitched to D2 (73.42 Hz) as root
        # Semitone offsets from D: 0=D, 1=D#, -2=C, 3=F, etc.
        # Format: list of semitone offsets, each element = 8th note
        
        # CHORUS bass pattern: 0*12*0*12*1*13*1*13*-2*10*-2*10*1*13*1*13*
        # (root + octave harmony throughout)
        self.chorus_bass_pattern = [
            0, 12, 0, 12, 1, 13, 1, 13, -2, 10, -2, 10, 1, 13, 1, 13
        ]
        
        # EPICNESS bass pattern: 0*__0*__1*__1*__-2*__-2*__1*__1*__ (plays twice)
        self.epicness_bass_pattern = [
            0, None, 0, None, 1, None, 1, None, -2, None, -2, None, 1, None, 1, None
        ]
        
        # MELODY patterns for G Minor scale - randomly generated each 2 bars
        # Format: list of (semitone_offset, duration_in_32nds) tuples or None for rest
        # Duration values: 1=32nd, 2=16th, 4=8th, 8=quarter
        # 2 bars at 140 BPM = 64 32nd notes total
        # Chorus: EXTREME patterns (dense, fast, varied rhythms, big jumps)
        # Dundundenden: SUBTLE patterns (sparser, longer notes)
        
        # Chorus extreme melody patterns - energetic but not too rapid
        # Using mostly 8th notes (4) and 16th notes (2), occasional 32nd pairs for accents
        self.chorus_melody_patterns = [
            # Pattern 1: Octave bounce - punchy 8th notes with occasional 16th fills
            [(0, 4), (12, 4), (0, 4), (7, 4), None, None, (5, 4), (12, 4),
             (0, 4), (-7, 4), (5, 4), (7, 4), None, None, (12, 4), (0, 4)],
            
            # Pattern 2: Syncopated groove - offbeat accents
            [None, None, (0, 4), None, None, (12, 4), None, None,
             (7, 4), None, None, (5, 4), None, None, (0, 4), (3, 4),
             None, None, (-7, 4), None, None, (5, 4), None, None,
             (12, 4), None, None, (7, 2), (5, 2), (0, 4), None, None],
            
            # Pattern 3: Call and response - phrase then answer
            [(0, 4), (3, 4), (5, 4), (7, 4), None, None, None, None,
             (12, 4), (7, 4), (5, 4), (0, 4), None, None, None, None,
             (-7, 4), (-5, 4), (0, 4), (3, 4), None, None, None, None,
             (5, 4), (7, 4), (12, 4), (0, 4), None, None, None, None],
            
            # Pattern 4: Wide leaps - dramatic jumps on 8ths
            [(0, 4), None, None, (12, 4), None, None, (-7, 4), None,
             None, (17, 4), None, None, (0, 4), None, None, (7, 4),
             None, None, (-5, 4), None, None, (12, 4), None, None,
             (3, 4), None, None, (15, 4), None, None, (0, 4), None],
            
            # Pattern 5: Driving 8ths with 16th turnarounds
            [(0, 4), (5, 4), (7, 4), (12, 4), (7, 2), (5, 2), (0, 4), None,
             (3, 4), (7, 4), (12, 4), (15, 4), (12, 2), (7, 2), (5, 4), None,
             (-7, 4), (-5, 4), (0, 4), (3, 4), (0, 2), (-2, 2), (-7, 4), None,
             (0, 4), (5, 4), (7, 4), (12, 4), None, None, (0, 4), None],
            
            # Pattern 6: Dotted rhythm feel
            [(0, 6), (5, 2), (7, 6), (12, 2), None, None, (0, 6), (3, 2),
             (-2, 6), (0, 2), (5, 6), (7, 2), None, None, (12, 4), (0, 4)],
            
            # Pattern 7: Sparse power hits
            [(0, 4), None, None, None, (12, 4), None, None, None,
             (7, 4), None, None, None, (-7, 4), None, None, None,
             (5, 4), None, None, None, (17, 4), None, None, None,
             (0, 4), None, None, None, (12, 4), None, None, None],
        ]
        
        # Dundundenden subtle melody patterns - sparser, more sustained, mostly 8th notes
        self.dundundenden_melody_patterns = [
            # Pattern 1: Long tones - very sparse
            [(0, 8), None, None, None, None, None, None, None,
             (7, 8), None, None, None, None, None, None, None,
             (5, 8), None, None, None, None, None, None, None,
             (0, 8), None, None, None, None, None, None, None],
            
            # Pattern 2: Breathing space - wide gaps
            [(0, 4), None, None, None, None, None, None, None,
             None, None, None, None, (7, 4), None, None, None,
             None, None, None, None, None, None, None, None,
             (-5, 4), None, None, None, None, None, (0, 4), None],
            
            # Pattern 3: Gentle pedal tone
            [(0, 8), None, None, None, (0, 8), None, None, None,
             (3, 4), (0, 4), None, None, (0, 8), None, None, None],
            
            # Pattern 4: Slow melodic phrase
            [(0, 8), None, None, None, (3, 8), None, None, None,
             (5, 8), None, None, None, (7, 8), None, None, None,
             (5, 8), None, None, None, (3, 8), None, None, None,
             (0, 8), None, None, None, None, None, None, None],
            
            # Pattern 5: Sparse offbeat hits
            [None, None, None, None, (0, 4), None, None, None,
             None, None, None, None, None, None, None, None,
             None, None, None, None, (5, 4), None, None, None,
             None, None, None, None, None, None, (0, 4), None],
            
            # Pattern 6: Single note drone
            [(0, 8), None, None, None, None, None, None, None,
             (0, 8), None, None, None, None, None, None, None,
             (1, 4), (0, 4), None, None, None, None, None, None,
             (0, 8), None, None, None, None, None, None, None],
        ]
        
        # Dundundenden EXTREME patterns for last 2 bars (same intensity as chorus)
        self.dundundenden_extreme_patterns = self.chorus_melody_patterns.copy()
        
        # AWESOMENESS pitch pattern (for LAST chorus only, center-right position)
        # Pattern: 0_3_12* 24* 1* 5 1 13*** -2_-2_10* 13* 8* 1* 13***0* 7* 12* 15* 17* 5* 1_5 1 -2* 1_29* 10* 1_5 1 20***
        # *** = 4th note (4 16ths), ** = 6th note (~3 16ths), * = 8th note (2 16ths), _ = 16th rest, no suffix = 16th note
        self.awesomeness_pattern_str = "0_3_12* 24* 1* 5 1 13*** -2_-2_10* 13* 8* 1* 13***0* 7* 12* 15* 17* 5* 1_5 1 -2* 1_29* 10* 1_5 1 20***"
        
        # EPICNESS pitch pattern (for epicness section only, top-left position)
        # Pattern: 0*__12*** 1*** 13*__10* 10_ 10*__13* 1* 13* 1*
        # *** = 4th note (4 16ths), * = 8th note (2 16ths), _ = 16th rest
        self.epicness_pitch_pattern_str = "0*__12*** 1*** 13*__10* 10_ 10*__13* 1* 13* 1*"
        
        # DUNDUNDENDEN chord progression:
        # Odd bars (1, 3, 5): D - D# (semitones 0, 1)
        # Even bars (2, 4, 6): C - D# (semitones -2, 1)
        # Bass can randomly pick notes from the current chord scale + octave jumps
        self.dundundenden_bass_chords = {
            'odd': [0, 1],    # D, D# for odd bars
            'even': [-2, 1],  # C, D# for even bars
        }
        
        # CHORD/PAD patterns for chorus - 3 or 4 simultaneous voices
        # Each chord is held for ~8 16ths (half bar)
        # Pattern format: list of (position_16th, [semitone_offsets], duration_16ths)
        # 4 chords per 2-bar pattern
        
        # Pattern A: 3 voices (basic triad)
        # 0****** 1****** -2****** 1******
        # 3****** 5****** 1****** 5******
        # 7****** 8****** 5****** 8******
        self.chord_pattern_3voice = [
            (0, [0, 3, 7], 8),      # Chord 1: D, F, A (D minor)
            (8, [1, 5, 8], 8),      # Chord 2: D#, G, G# 
            (16, [-2, 1, 5], 8),    # Chord 3: C, D#, G
            (24, [1, 5, 8], 8),     # Chord 4: D#, G, G#
        ]
        
        # Pattern B: 4 voices (extended chord)
        # 0****** 1****** -2****** 1******
        # 3****** 5****** 1****** 5******
        # 7****** 8****** 5****** 8******
        # 10***** 12****** 8****** 12******
        self.chord_pattern_4voice = [
            (0, [0, 3, 7, 10], 8),     # Chord 1 + 10th
            (8, [1, 5, 8, 12], 8),     # Chord 2 + octave
            (16, [-2, 1, 5, 8], 8),    # Chord 3 + 8
            (24, [1, 5, 8, 12], 8),    # Chord 4 + octave
        ]

        # Zorammi chord line patterns (per voice) - semitone offsets from D
        # Chorus: leading rests per 8th; Dundundenden: immediate hit at bar start
        self.zorammi_chorus_lines = [
            "__0*__0*__1*__1*__-2*__-2*__1*__1*",
            "__3*__3*__5*__5*__1*__1*__5*__5*",
            "__7*__7*__8*__8*__5*__5*__8*__8*",
        ]
        self.zorammi_dundun_lines = [
            "0*__0*__1*__1*__-2*__-2*__1*__1*__",
            "3*__3*__5*__5*__1*__1*__5*__5*__",
            "7*__7*__8*__8*__5*__5*__8*__8*__",
        ]
    
    def expand_pattern(self, pattern_str: str, pattern_length: int) -> List[str]:
        """Expand a pattern string into individual events (drums/bass)."""
        expanded = []
        for char in pattern_str:
            if char in ['_', '.']:
                expanded.append(None)
            else:
                expanded.append(char)

        if len(expanded) < pattern_length:
            expanded = (expanded * (pattern_length // len(expanded) + 1))[:pattern_length]
        elif len(expanded) > pattern_length:
            expanded = expanded[:pattern_length]
        return expanded
    
    def generate_drum_pattern(self, pattern_name: str, num_bars: int) -> Dict[str, List]:
        """Generate a drum pattern based on the pattern name."""
        if pattern_name not in PATTERNS:
            pattern_name = 'chorus'  # Default to chorus if pattern not found
            
        pattern = PATTERNS[pattern_name]
        
        # Calculate the number of 16th notes in the pattern
        sixteenths_per_bar = 16
        total_sixteenths = num_bars * sixteenths_per_bar
        
        # Expand the drum pattern
        drum_pattern = self.expand_pattern(pattern['drums'], total_sixteenths)
        
        return {
            'kick': [i for i, x in enumerate(drum_pattern) if x == 'K'],
            'snare': [i for i, x in enumerate(drum_pattern) if x == 'S'],
            'hihat': [i for i, x in enumerate(drum_pattern) if x == 'H']
        }
    
    def generate_bass_pattern(self, pattern_name: str, num_bars: int, num_samples: int) -> List[Tuple[int, int]]:
        """Generate a bass pattern with sample indices."""
        if pattern_name not in PATTERNS or 'bass' not in PATTERNS[pattern_name]:
            # Default bass pattern: one note per beat
            pattern = "1_2_3_4_" * num_bars
        else:
            pattern = PATTERNS[pattern_name]['bass']
        
        # Expand the pattern
        expanded = self.expand_pattern(pattern, num_bars * 8)  # 8th notes
        
        # Convert to (position, sample_index) tuples, wrapping sample indices
        result = []
        for pos, sample_idx in enumerate(expanded):
            if sample_idx is not None:
                wrapped_idx = sample_idx % num_samples if num_samples > 0 else 0
                result.append((pos, wrapped_idx))
        
        return result
    
    def _parse_vocal_pattern(
        self,
        pattern: str,
        max_len_16ths: int,
        num_samples: int,
        allowed_indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, int]]:
        """
        Parse vocal pattern into (position_16th, sample_idx, duration_16ths).
        Supports '*' after a digit to make it an 8th note (2x 16th).
        '_' = rest (1x 16th).
        """
        events = []
        pos = 0
        i = 0
        while i < len(pattern) and pos < max_len_16ths:
            ch = pattern[i]
            if ch == '_':
                pos += 1
                i += 1
                continue
            if ch.isdigit():
                dur = 1
                if i + 1 < len(pattern) and pattern[i + 1] == '*':
                    dur = 2
                    i += 1  # skip '*'
                sample_idx = int(ch) - 1
                if allowed_indices:
                    if not allowed_indices:
                        i += 1
                        pos += dur
                        continue
                    wrapped = allowed_indices[sample_idx % len(allowed_indices)]
                else:
                    wrapped = sample_idx % num_samples if num_samples > 0 else 0
                events.append((pos, wrapped, dur))
                pos += dur
                i += 1
                continue
            # Any other char acts as rest
            pos += 1
            i += 1
        return events

    def _generate_vocal_variation(self, base: str, num_bars: int, rng: random.Random) -> str:
        """Return a mutated pattern string of the same length, staying on-grid."""
        max_len = num_bars * 16
        tokens = list(base)
        tokens = (tokens * (max_len // len(tokens) + 1))[:max_len]
        digits = ['1', '2', '3']
        bar_len = 16
        for bar in range(num_bars):
            start = bar * bar_len
            end = start + bar_len
            # Limit mutations per bar to keep shape recognizable
            max_mutations = 3
            mutations = 0
            for i in range(start, end):
                if mutations >= max_mutations:
                    break
                t = tokens[i]
                beat_pos = i % 4
                if t in ['_', '.']:
                    # Very low chance to fill a rest on downbeats only
                    if beat_pos == 0 and rng.random() < 0.08:
                        tokens[i] = rng.choice(digits)
                        mutations += 1
                    continue
                if t.isdigit():
                    # Small chance to swap digit (keep 1/2/3 palette)
                    if rng.random() < 0.15:
                        tokens[i] = rng.choice(digits)
                        mutations += 1
                    # Rare 8th note extension on strong beats if space
                    if beat_pos in (0, 2) and rng.random() < 0.06 and i + 1 < len(tokens) and tokens[i + 1] == '_':
                        tokens[i] = tokens[i] + '*'
                        tokens[i + 1] = ''  # mark consumed
                        mutations += 1
        tokens = [t for t in tokens if t != '']
        if len(tokens) < max_len:
            tokens = (tokens * (max_len // len(tokens) + 1))[:max_len]
        elif len(tokens) > max_len:
            tokens = tokens[:max_len]
        return "".join(tokens)

    def generate_vocal_pattern(
        self,
        pattern_name: str,
        num_bars: int,
        num_samples: int,
        allowed_indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, int]]:
        """Generate a vocal chop pattern with sample indices (optionally restricted)."""
        base_patterns = self.vocal_patterns.get(pattern_name, [])
        if not base_patterns:
            base_patterns = ["".join(random.choice(["1", "2", "_"]) for _ in range(num_bars * 16))]

        max_len = num_bars * 16
        events: List[Tuple[int, int, int]] = []

        rng = random.Random(self.variation_seed)
        allow_variation = self.variation_enabled and pattern_name in {"chorus", "dundundenden"}

        # Special case: epicness uses multiple lines simultaneously, no variation
        if pattern_name == "epicness":
            for base_pat in base_patterns:
                repeated = (base_pat * (max_len // len(base_pat) + 1))[:max_len]
                events.extend(self._parse_vocal_pattern(repeated, max_len, num_samples, allowed_indices))
            return events

        # Use the first base pattern, expanded to section length
        base = (base_patterns[0] * (max_len // len(base_patterns[0]) + 1))[:max_len]
        variant = self._generate_vocal_variation(base, num_bars, rng) if allow_variation else base

        # Build bar-wise blocks to stay on-grid; first bar always base
        final_tokens: List[str] = []
        for bar in range(num_bars):
            start = bar * 16
            end = start + 16
            block_base = base[start:end]
            block_variant = variant[start:end]
            # For chorus/dundundenden: bar 0 base; others maybe variant but not all
            if allow_variation and bar > 0 and rng.random() < 0.4:
                chosen = block_variant
            else:
                chosen = block_base
            final_tokens.append(chosen)

        final_pattern = "".join(final_tokens)
        final_pattern = (final_pattern * (max_len // len(final_pattern) + 1))[:max_len]

        events.extend(self._parse_vocal_pattern(final_pattern, max_len, num_samples, allowed_indices))

        return events
    
    def generate_chord_progression(self, pattern_name: str, num_bars: int) -> List[str]:
        """Generate a chord progression for the pattern."""
        if pattern_name not in PATTERNS or 'chords' not in PATTERNS[pattern_name]:
            # Default to C major if no chords specified
            return ['C3'] * num_bars
        
        chords = PATTERNS[pattern_name]['chords']
        
        # Repeat the chord progression to fill the number of bars
        return [chords[i % len(chords)] for i in range(num_bars)]

    def generate_bass_instrument_pattern(self, pattern_name: str, num_bars: int) -> List[Tuple[int, int, int]]:
        """
        Generate bass instrument events for a section.
        Returns list of (position_16th, semitone_offset, duration_16ths).
        
        Bass is pitched to D2 (73.42 Hz) as root. Semitone offsets shift from there.
        """
        import random
        
        eighth_note_16ths = 2  # each 8th note = 2 sixteenths
        events = []
        
        if pattern_name == 'chorus':
            # Chorus: loop the chorus_bass_pattern (16 eighths = 2 bars)
            pattern = self.chorus_bass_pattern
            bars_per_pattern = 2
            
            num_loops = (num_bars + 1) // bars_per_pattern
            for loop_idx in range(num_loops):
                loop_start_16th = loop_idx * bars_per_pattern * 16
                
                for i, semitone in enumerate(pattern):
                    if semitone is not None:
                        position_16th = loop_start_16th + (i * eighth_note_16ths)
                        if position_16th >= num_bars * 16:
                            break
                        events.append((position_16th, semitone, eighth_note_16ths))
            
            print(f"  [Bass] chorus: {len(events)} events over {num_bars} bars")
            
        elif pattern_name == 'epicness':
            # Epicness: play the pattern twice (each pattern = 2 bars, total 4 bars)
            pattern = self.epicness_bass_pattern
            bars_per_pattern = 2
            
            # Play twice
            for repeat in range(2):
                repeat_start_16th = repeat * bars_per_pattern * 16
                
                for i, semitone in enumerate(pattern):
                    if semitone is not None:
                        position_16th = repeat_start_16th + (i * eighth_note_16ths)
                        if position_16th >= num_bars * 16:
                            break
                        events.append((position_16th, semitone, eighth_note_16ths))
            
            print(f"  [Bass] epicness: {len(events)} events over {num_bars} bars")
            
        elif pattern_name == 'dundundenden':
            # Dundundenden: random patterns following chord progression
            # Odd bars: D - D# | Even bars: C - D#
            # Generate random 8th note patterns within the chord scale
            
            for bar in range(num_bars):
                bar_start_16th = bar * 16
                is_odd_bar = (bar % 2 == 0)  # 0-indexed, so bar 0 = "bar 1" = odd
                
                chord_notes = self.dundundenden_bass_chords['odd' if is_odd_bar else 'even']
                
                # Generate a random pattern for this bar (8 eighth notes = 16 sixteenths)
                # Randomly pick rhythm: some notes, some rests
                for eighth_idx in range(8):
                    position_16th = bar_start_16th + (eighth_idx * eighth_note_16ths)
                    
                    # ~60% chance to play a note on each 8th
                    if random.random() < 0.6:
                        # Pick a note from chord, maybe with octave jump
                        base_note = random.choice(chord_notes)
                        
                        # 30% chance to jump up an octave (+12 semitones)
                        if random.random() < 0.3:
                            base_note += 12
                        
                        events.append((position_16th, base_note, eighth_note_16ths))
            
            print(f"  [Bass] dundundenden: {len(events)} events over {num_bars} bars (random)")
        
        return events

    def generate_pitch_pattern(self, pattern_name: str, num_bars: int) -> List[Tuple[int, int, int]]:
        """
        Generate pitch sample events for a section.
        Returns list of (position_16th, semitone_offset, duration_16ths).
        
        For chorus and dundundenden: randomly selects a pattern every 2 bars.
        Each element in pattern = 8th note (2 sixteenths).
        """
        import random
        
        # Get the appropriate pattern list for this section
        if pattern_name == 'chorus':
            pattern_list = self.chorus_pitch_patterns
        elif pattern_name == 'dundundenden':
            pattern_list = self.dundundenden_pitch_patterns
        else:
            return []  # No pitch patterns for other sections
        
        eighth_note_16ths = 2  # each 8th note = 2 sixteenths
        bars_per_pattern = 2  # patterns are 2 bars long (16 eighth notes)
        eighths_per_pattern = 16  # 16 eighth notes per 2-bar pattern
        
        events = []
        
        # Process 2 bars at a time, randomly selecting a pattern each time
        num_2bar_chunks = (num_bars + 1) // 2  # Round up
        
        for chunk_idx in range(num_2bar_chunks):
            # Randomly select a pattern for this 2-bar chunk
            selected_pattern = random.choice(pattern_list)
            
            # Calculate starting position for this chunk
            chunk_start_16th = chunk_idx * bars_per_pattern * 16
            
            # Generate events for this pattern
            for i, semitone in enumerate(selected_pattern):
                if semitone is not None:  # Skip rests (None values)
                    position_16th = chunk_start_16th + (i * eighth_note_16ths)
                    
                    # Don't exceed the section length
                    if position_16th >= num_bars * 16:
                        break
                    
                    events.append((position_16th, semitone, eighth_note_16ths))
            
            # Log which pattern was selected
            pattern_idx = pattern_list.index(selected_pattern)
            print(f"  [Pitch] {pattern_name} bars {chunk_idx*2+1}-{chunk_idx*2+2}: pattern #{pattern_idx+1}")
        
        return events

    def _parse_pitch_pattern_string(self, pattern_str: str) -> List[Tuple[int, int, int]]:
        """
        Parse a pitch pattern string into events.
        Returns list of (position_16th, semitone, duration_16ths).
        
        Notation:
        - Number (possibly negative) = semitone offset
        - *** after number = 4th note (4 16ths)
        - ** after number = 6th note (3 16ths)  
        - * after number = 8th note (2 16ths)
        - No suffix = 16th note (1 16th)
        - _ = 16th rest
        - Space = visual separator (separates tokens but doesn't advance position)
        """
        events = []
        pos = 0
        i = 0
        pattern = pattern_str  # Keep spaces to prevent merging numbers!
        
        while i < len(pattern):
            ch = pattern[i]
            
            # Skip spaces (visual separators only, don't merge adjacent tokens)
            if ch == ' ':
                i += 1
                continue
            
            # Handle rest
            if ch == '_':
                pos += 1
                i += 1
                continue
            
            # Handle number (semitone) - may be negative
            if ch == '-' or ch.isdigit():
                # Parse the full number
                num_str = ''
                if ch == '-':
                    num_str = '-'
                    i += 1
                
                while i < len(pattern) and pattern[i].isdigit():
                    num_str += pattern[i]
                    i += 1
                
                if not num_str or num_str == '-':
                    continue
                    
                semitone = int(num_str)
                
                # Check for duration suffix
                dur = 1  # Default: 16th note
                if i < len(pattern) and pattern[i] == '*':
                    star_count = 0
                    while i < len(pattern) and pattern[i] == '*':
                        star_count += 1
                        i += 1
                    
                    if star_count >= 3:
                        dur = 4  # 4th note (quarter)
                    elif star_count == 2:
                        dur = 3  # 6th note (dotted 8th, ~3 16ths)
                    else:
                        dur = 2  # 8th note
                
                events.append((pos, semitone, dur))
                pos += dur
                continue
            
            # Skip any other character
            i += 1
        
        # Debug: print parsed pattern
        print(f"  [Pattern] Parsed {len(events)} events, total length: {pos} 16ths ({pos/16:.1f} bars)")
        
        return events

    def generate_awesomeness_pattern(self, num_bars: int, is_last_chorus: bool = False) -> List[Tuple[int, int, int]]:
        """
        Generate awesomeness pitch pattern events for LAST chorus only.
        Returns list of (position_16th, semitone_offset, duration_16ths).
        
        Only generates events if is_last_chorus is True.
        Pattern plays ONLY in the LAST 4 BARS of the chorus (starting at bar 5).
        """
        if not is_last_chorus:
            return []
        
        # Awesomeness only plays in the last 4 bars of an 8-bar chorus
        # So it starts at bar 5 (position 64 in 16ths: 4 bars * 16 = 64)
        if num_bars < 8:
            # If chorus is less than 8 bars, don't play awesomeness
            print(f"  [Awesomeness] Skipping - chorus only has {num_bars} bars (need 8)")
            return []
        
        # Parse the awesomeness pattern string
        pattern_events = self._parse_pitch_pattern_string(self.awesomeness_pattern_str)
        
        # Calculate pattern length in 16ths
        if pattern_events:
            pattern_length = max(pos + dur for pos, _, dur in pattern_events)
        else:
            pattern_length = 64  # 4 bars default
        
        # Awesomeness starts at bar 5 (16th position 64) and plays for last 4 bars
        start_offset = 64  # Bar 5 starts at 16th note 64 (4 bars * 16)
        last_4_bars_16ths = 64  # 4 bars * 16 16th notes
        
        events = []
        current_offset = start_offset
        
        while current_offset < start_offset + last_4_bars_16ths:
            for pos, semitone, dur in pattern_events:
                absolute_pos = current_offset + pos
                if absolute_pos >= start_offset + last_4_bars_16ths:
                    break
                events.append((absolute_pos, semitone, dur))
            current_offset += pattern_length
        
        print(f"  [Awesomeness] Generated {len(events)} events for last 4 bars of chorus (bars 5-8)")
        return events

    def generate_epicness_pitch_pattern(self, num_bars: int) -> List[Tuple[int, int, int]]:
        """
        Generate epicness pitch pattern events for epicness section only.
        Returns list of (position_16th, semitone_offset, duration_16ths).
        """
        # Parse the epicness pitch pattern string
        pattern_events = self._parse_pitch_pattern_string(self.epicness_pitch_pattern_str)
        
        # Calculate pattern length in 16ths
        if pattern_events:
            pattern_length = max(pos + dur for pos, _, dur in pattern_events)
        else:
            pattern_length = 32  # 2 bars default
        
        # Loop pattern to fill the section
        events = []
        total_16ths = num_bars * 16
        current_offset = 0
        
        while current_offset < total_16ths:
            for pos, semitone, dur in pattern_events:
                absolute_pos = current_offset + pos
                if absolute_pos >= total_16ths:
                    break
                events.append((absolute_pos, semitone, dur))
            current_offset += pattern_length
        
        print(f"  [Epicness Pitch] Generated {len(events)} events for epicness ({num_bars} bars)")
        return events

    def generate_chord_pattern(self, pattern_name: str, num_bars: int) -> List[Tuple[int, List[int], int]]:
        """
        Generate chord/pad events for chorus sections.
        Returns list of (position_16th, [semitone_offsets], duration_16ths).
        
        Each chord event has multiple simultaneous voices that should be layered.
        Randomly picks between 3-voice and 4-voice patterns each generation.
        """
        import random
        
        # Only generate chords for chorus sections
        if pattern_name != 'chorus':
            return []
        
        # Randomly choose between 3-voice and 4-voice pattern (flip a coin)
        use_4_voice = random.choice([True, False])
        base_pattern = self.chord_pattern_4voice if use_4_voice else self.chord_pattern_3voice
        voice_count = 4 if use_4_voice else 3
        
        print(f"  [Chords] Using {voice_count}-voice pattern for chorus")
        
        events = []
        pattern_length_16ths = 32  # 2 bars = 32 sixteenths
        
        # Loop the pattern to fill all bars
        num_2bar_chunks = (num_bars + 1) // 2
        
        for chunk_idx in range(num_2bar_chunks):
            chunk_start_16th = chunk_idx * pattern_length_16ths
            
            for pos, semitones, dur in base_pattern:
                absolute_pos = chunk_start_16th + pos
                
                # Don't exceed section length
                if absolute_pos >= num_bars * 16:
                    break
                
                events.append((absolute_pos, semitones, dur))
        
        print(f"  [Chords] Generated {len(events)} chord events for {num_bars} bars")
        return events

    def _parse_zorammi_lines(self, lines: List[str]) -> Tuple[List[Tuple[int, int, int]], int]:
        """Parse per-voice lines into events and return pattern length in 16ths."""
        voice_events: List[Tuple[int, int, int]] = []
        max_len = 0
        for line in lines:
            parsed = self._parse_pitch_pattern_string(line)
            voice_events.extend(parsed)
            if parsed:
                max_len = max(max_len, max(pos + dur for pos, _, dur in parsed))
        return voice_events, max_len

    def generate_zorammi_chords_pattern(self, section_name: str, num_bars: int) -> List[Tuple[int, List[int], int]]:
        """
        Generate Zorammi chord events for chorus and dundundenden.
        Returns list of (position_16th, [semitone_offsets], duration_16ths).
        """
        if section_name not in {"chorus", "dundundenden"}:
            return []

        lines = self.zorammi_chorus_lines if section_name == "chorus" else self.zorammi_dundun_lines
        parsed, pattern_len = self._parse_zorammi_lines(lines)
        if pattern_len == 0:
            return []
        # Add an 8th-note rest gap between loops in dundundenden
        if section_name == "dundundenden":
            pattern_len += 2  # two 16ths = one 8th note

        # Group events by position within pattern
        grouped: Dict[int, Tuple[List[int], int]] = {}
        for pos, semitone, dur in parsed:
            if pos not in grouped:
                grouped[pos] = ([], dur)
            grouped[pos][0].append(semitone)
            # keep duration as max in case of mismatch
            grouped[pos] = (grouped[pos][0], max(grouped[pos][1], dur))

        total_16ths = num_bars * 16
        events: List[Tuple[int, List[int], int]] = []
        loop_start = 0
        while loop_start < total_16ths:
            for pos, (semitones, dur) in grouped.items():
                absolute_pos = loop_start + pos
                if absolute_pos >= total_16ths:
                    break
                events.append((absolute_pos, semitones.copy(), dur))
            loop_start += pattern_len

        events.sort(key=lambda x: x[0])
        print(f"  [Zorammi] {section_name}: {len(events)} events over {num_bars} bars (pattern len {pattern_len} 16ths)")
        return events

    def generate_melody_pattern(self, pattern_name: str, num_bars: int) -> List[Tuple[int, int, int]]:
        """
        Generate melody sample events for a section using G Minor scale.
        Returns list of (position_32nds, semitone_offset, duration_32nds).
        
        Pattern format: list of (semitone, duration_32nds) tuples or None for rest
        Duration values: 1=32nd, 2=16th, 4=8th, 8=quarter
        
        - Chorus: EXTREME patterns (dense, fast, wide intervals, varied rhythms)
        - Dundundenden: SUBTLE patterns, EXCEPT last 2 bars which are EXTREME
        """
        import random
        
        # Only generate melody for chorus and dundundenden
        if pattern_name not in ['chorus', 'dundundenden']:
            return []
        
        bars_per_pattern = 2  # patterns are 2 bars long
        thirty_seconds_per_bar = 32  # 32 32nd notes per bar at 4/4
        
        events = []
        
        # Process 2 bars at a time
        num_2bar_chunks = (num_bars + 1) // 2
        
        for chunk_idx in range(num_2bar_chunks):
            chunk_start_32nd = chunk_idx * bars_per_pattern * thirty_seconds_per_bar
            
            # Determine which pattern set to use
            if pattern_name == 'chorus':
                # Chorus: always use extreme patterns
                pattern_list = self.chorus_melody_patterns
                intensity = "EXTREME"
            elif pattern_name == 'dundundenden':
                # Dundundenden: subtle for first chunks, extreme for last 2 bars
                is_last_2_bars = (chunk_idx == num_2bar_chunks - 1)
                if is_last_2_bars:
                    pattern_list = self.dundundenden_extreme_patterns
                    intensity = "EXTREME"
                else:
                    pattern_list = self.dundundenden_melody_patterns
                    intensity = "subtle"
            else:
                continue
            
            # Randomly select a pattern
            selected_pattern = random.choice(pattern_list)
            
            # Generate events for this pattern
            current_pos_32nd = 0
            note_count = 0
            for item in selected_pattern:
                if item is None:
                    # Rest - advance by 1 32nd note (smallest unit for rests)
                    current_pos_32nd += 1
                else:
                    semitone, duration_32nds = item
                    absolute_pos_32nd = chunk_start_32nd + current_pos_32nd
                    
                    # Don't exceed section length (convert to 32nds)
                    if absolute_pos_32nd >= num_bars * thirty_seconds_per_bar:
                        break
                    
                    events.append((absolute_pos_32nd, semitone, duration_32nds))
                    current_pos_32nd += duration_32nds
                    note_count += 1
            
            print(f"  [Melody] {pattern_name} bars {chunk_idx*2+1}-{chunk_idx*2+2}: {intensity} ({note_count} notes)")
        
        return events

    def create_section(self, section_name: str, num_bars: int, samples: Dict[str, list], is_last_chorus: bool = False) -> Dict:
        """Create a complete section with all elements."""
        # Initialize section
        section = {
            'name': section_name,
            'length_bars': num_bars,
            'drums': {},
            'bass': [],
            'chords': [],
            'vocals': [],
            'pitch': [],  # Pitch sample events (position, semitone, duration)
            'melody': [],  # Melody sample events (position, semitone, duration) - G Minor scale
            'bass_instrument': [],  # Bass instrument events (position, semitone, duration)
            'chord_pads': [],  # Chord pad events (position, [semitones], duration) - chorus only
            'awesomeness': [],  # Awesomeness events (position, semitone, duration) - last chorus only
            'epicness_pitch': [],  # Epicness pitch events (position, semitone, duration) - epicness only
            'zorammi_chords': [],  # Zorammi chord events (position, [semitones], duration)
        }
        
        # Generate drum pattern
        drum_pattern = self.generate_drum_pattern(section_name, num_bars)
        section['drums'] = drum_pattern
        
        # Generate vocal pattern if we have vocal samples
        if 'vocal' in samples and samples['vocal']:
            # Pick up to 3 strongest snippets (enough for digits 1-3)
            allowed_indices = None
            vocals = samples['vocal']
            sorted_indices = sorted(
                range(len(vocals)),
                key=lambda i: vocals[i]['analysis'].get('rms', 0),
                reverse=True
            )
            max_needed = 3  # digits 1,2,3 appear in patterns
            allowed_indices = sorted_indices[:max_needed] if sorted_indices else []
            num_vocal_samples = len(allowed_indices) if allowed_indices else len(vocals)

            vocal_pattern = self.generate_vocal_pattern(
                section_name,
                num_bars,
                num_vocal_samples,
                allowed_indices=allowed_indices
            )
            section['vocals'] = vocal_pattern
        
        # Generate pitch pattern (only for chorus and dundundenden)
        section['pitch'] = self.generate_pitch_pattern(section_name, num_bars)
        
        # Generate melody pattern (only for chorus and dundundenden)
        section['melody'] = self.generate_melody_pattern(section_name, num_bars)
        
        # Generate bass instrument pattern (chorus, epicness, dundundenden)
        section['bass_instrument'] = self.generate_bass_instrument_pattern(section_name, num_bars)
        
        # Generate chord pad pattern (chorus only)
        section['chord_pads'] = self.generate_chord_pattern(section_name, num_bars)
        
        # Generate awesomeness pattern (LAST chorus only)
        if section_name == 'chorus' and is_last_chorus:
            section['awesomeness'] = self.generate_awesomeness_pattern(num_bars, is_last_chorus=True)
        
        # Generate epicness pitch pattern (epicness section only)
        if section_name == 'epicness':
            section['epicness_pitch'] = self.generate_epicness_pitch_pattern(num_bars)

        # Generate Zorammi chords (chorus + dundundenden) when style enabled
        if self.zorammi_style and section_name in {"chorus", "dundundenden"}:
            section['zorammi_chords'] = self.generate_zorammi_chords_pattern(section_name, num_bars)

        return section
    
    def create_song_structure(self, samples: Dict[str, list], 
                               remix_length: int = 6) -> List[Dict]:
        """
        Create a complete song structure with multiple sections.
        
        Args:
            samples: Dict of categorized samples
            remix_length: How much of the remix to generate (1-6 scale):
                1 = Intro only (2 bars)
                2 = Intro + first chorus (6 bars)
                3 = Intro + chorus + dundundenden (12 bars)
                4 = Intro + chorus + dundundenden + chorus (20 bars)
                5 = Intro + chorus + dundundenden + chorus + epicness (24 bars)
                6 = Full remix (32 bars)
        
        Returns:
            List of section dictionaries
        """
        # Full structure definition
        full_structure = [
            ("intro", 2),        # 2 bars  - level 1
            ("chorus", 4),       # 4 bars  - level 2
            ("dundundenden", 6), # 6 bars  - level 3
            ("chorus", 8),       # 8 bars  - level 4
            ("epicness", 4),     # 4 bars  - level 5
            ("chorus", 8),       # final 8 bars - level 6
        ]
        
        # Clamp remix_length to valid range
        remix_length = max(1, min(6, remix_length))
        
        # Take only the sections up to the specified length
        structure = full_structure[:remix_length]
        
        # Calculate total bars for info
        total_bars = sum(bars for _, bars in structure)
        section_names = [name for name, _ in structure]
        print(f"  [Structure] Generating {remix_length}/6 sections: {', '.join(section_names)} ({total_bars} bars)")
        
        # Find the last chorus index to mark it for awesomeness pattern
        last_chorus_idx = -1
        for idx, (name, _) in enumerate(structure):
            if name == 'chorus':
                last_chorus_idx = idx
        
        # Generate each section
        song = []
        for idx, (section_name, num_bars) in enumerate(structure):
            is_last_chorus = (section_name == 'chorus' and idx == last_chorus_idx)
            section = self.create_section(section_name, num_bars, samples, is_last_chorus=is_last_chorus)
            song.append(section)
        
        return song
    
    @staticmethod
    def get_remix_length_info() -> Dict[int, Dict]:
        """Get info about each remix length level for UI display."""
        return {
            1: {"name": "Intro Only", "bars": 2, "sections": ["intro"]},
            2: {"name": "Intro + Chorus", "bars": 6, "sections": ["intro", "chorus"]},
            3: {"name": "Half Remix", "bars": 12, "sections": ["intro", "chorus", "dundundenden"]},
            4: {"name": "Extended", "bars": 20, "sections": ["intro", "chorus", "dundundenden", "chorus"]},
            5: {"name": "Almost Full", "bars": 24, "sections": ["intro", "chorus", "dundundenden", "chorus", "epicness"]},
            6: {"name": "Full Remix", "bars": 32, "sections": ["intro", "chorus", "dundundenden", "chorus", "epicness", "chorus"]},
        }
