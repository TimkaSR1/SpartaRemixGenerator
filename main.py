import os
import sys
import argparse
from pathlib import Path
import shutil
import random

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from audio_processor import AudioProcessor
from pattern_generator import PatternGenerator
from arranger import Arranger
from config import ensure_directories, TEMP_DIR, SAMPLE_DIR, OUTPUT_DIR, RESOURCE_DIR

def clear_directory(directory):
    """Clear all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def process_video(video_path: str, output_path: str = None, bpm: int = 140, 
                  quote_start: float = None, quote_end: float = None,
                  remix_length: int = 6, instrument_toggles: dict = None,
                  progress_callback=None) -> str:
    """
    Process a video file and generate a Sparta Remix.
    
    Args:
        video_path: Path to input video
        output_path: Path for output file
        bpm: Beats per minute
        quote_start: Start time in seconds for quote selection (optional)
        quote_end: End time in seconds for quote selection (optional)
        remix_length: How much of the remix to generate (1-6 scale):
            1 = Intro only (2 bars)
            2 = Intro + first chorus (6 bars)
            3 = Intro + chorus + dundundenden (12 bars)
            4 = Intro + chorus + dundundenden + chorus (20 bars)
            5 = Intro + chorus + dundundenden + chorus + epicness (24 bars)
            6 = Full remix (32 bars)
        instrument_toggles: Dict of instrument enable/disable states (optional)
        progress_callback: Function to call with progress updates (optional)
    """
    def report(msg, pct: int = None):
        if pct is not None:
            pct = max(0, min(100, int(pct)))
            tagged = f"{pct}% {msg}"
            print(tagged)
            if progress_callback:
                progress_callback(tagged)
            return

        print(msg)
        if progress_callback:
            progress_callback(msg)
    
    report("Starting Sparta Remix generation...", 0)
    if quote_start is not None and quote_end is not None:
        print(f"Quote selection: {quote_start:.2f}s to {quote_end:.2f}s")
    
    # Ensure all directories exist
    ensure_directories()

    video_output_path = None
    audio_output_path = None
    if output_path:
        out_p = Path(output_path)
        out_suffix = out_p.suffix.lower()
        if out_suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            video_output_path = str(out_p)
            audio_output_path = str(out_p.with_suffix(".wav"))
        else:
            audio_output_path = str(out_p)
            video_output_path = str(out_p.with_suffix(".mp4"))
    else:
        audio_output_path = str(OUTPUT_DIR / "sparta_remix.wav")
        video_output_path = str(OUTPUT_DIR / "sparta_remix.mp4")
    
    # Clear temporary directories
    print("Cleaning up temporary directories...")
    clear_directory(TEMP_DIR)
    SAMPLE_DIR.mkdir(exist_ok=True)
    
    # Set instrument toggles (default: all on except Zorammi chords/style)
    if instrument_toggles is None:
        instrument_toggles = {
            'kick': True, 'snare': True, 'hihat': True, 'bass': True,
            'pitch': True, 'melody': True, 'chords': True, 'vocals': True,
            'awesomeness': True, 'epicness_pitch': True, 'zorammi_chords': False,
            'zorammi_style': False
        }

    zorammi_on = instrument_toggles.get('zorammi_style', False) or instrument_toggles.get('zorammi_chords', False)

    # Enforce toggle rules per request
    if zorammi_on:
        # ON: only vocals, kick, snare, hihat, bass, zorammi_chords
        allowed = {'kick', 'snare', 'hihat', 'bass', 'vocals', 'zorammi_chords'}
        for k in list(instrument_toggles.keys()):
            if k == 'zorammi_style':
                continue
            instrument_toggles[k] = k in allowed
        # ensure zorammi flags on
        instrument_toggles['zorammi_chords'] = True
        instrument_toggles['zorammi_style'] = True
    else:
        # OFF: everything ON except zorammi_chords
        for k in list(instrument_toggles.keys()):
            if k == 'zorammi_style':
                instrument_toggles[k] = False
            elif k == 'zorammi_chords':
                instrument_toggles[k] = False
            else:
                instrument_toggles[k] = True

    # Initialize components
    audio_processor = AudioProcessor()
    pattern_generator = PatternGenerator(bpm=bpm, zorammi_style=zorammi_on)
    arranger = Arranger(bpm=bpm)
    arranger.zorammi_style = zorammi_on
    # Double-ensure flag sticks
    pattern_generator.zorammi_style = zorammi_on
    
    arranger.instrument_toggles = instrument_toggles
    print(f"Instrument toggles: {instrument_toggles}")
    
    try:
        # Step 1: Extract audio from video (with quote selection if specified)
        report("📥 Extracting audio from video...", 5)
        # Use full audio for all processing; quote window only filters chorus sample selection
        audio_path = audio_processor.extract_audio_from_video(video_path)
        
        # Step 2: Process audio and extract samples
        report("🔍 Extracting samples from audio...", 15)
        samples = audio_processor.process_audio_file(audio_path)
        
        # Print sample statistics
        print("\nSample extraction complete:")
        for category, sample_list in samples.items():
            print(f"- {category}: {len(sample_list)} samples")
        
        # Determine chorus quote window (only used to filter chorus vocals)
        chorus_window = None
        if quote_start is not None and quote_end is not None and quote_end > quote_start:
            chorus_window = (quote_start, quote_end)
            print(f"\nChorus quote window set to {quote_start:.2f}s - {quote_end:.2f}s (vocals only)")
        
        # Step 2b: Select pure vowel samples for chorus within window (no consonants)
        report("🎤 Selecting vocal samples...", 25)
        pure_vowels = audio_processor.get_pure_vowel_samples(samples, top_n=3, window=chorus_window)
        
        # Replace the vocal samples with pure vowels for better chorus sound
        if pure_vowels:
            samples['vocal'] = pure_vowels
            print(f"Using {len(pure_vowels)} pure vowel samples for chorus")
        
        # Step 2c: Find and pitch-correct the best sample for "pitch" instrument
        # Exclude the pure vowel samples to get variety
        print("\nFinding best pitch sample (sustained 'aaahh' sound)...")
        used_vocal_paths = [s['path'] for s in pure_vowels]
        
        best_pitch_sample, is_fallback = audio_processor.get_best_pitch_sample(
            samples, exclude_paths=used_vocal_paths, random_pick=True
        )
        pitch_corrected_path = None
        if best_pitch_sample:
            # Log transient quality
            transient_score = best_pitch_sample.get('transient_score', 0.5)
            trim_samples = best_pitch_sample.get('trim_samples', 0)
            print(f"Best pitch sample: {Path(best_pitch_sample['path']).name}")
            print(f"  Transient score: {transient_score:.2f} (1.0=sharp, 0.0=slow)")
            
            # First, trim slow attack if needed
            sample_to_process = best_pitch_sample['path']
            if trim_samples > 0 and transient_score < 0.7:
                trimmed_path = str(TEMP_DIR / "pitch_trimmed.wav")
                sample_to_process = audio_processor.trim_sample_attack(
                    best_pitch_sample['path'], trimmed_path, trim_samples
                )
            
            if is_fallback:
                # Use fallback processing: over-stretch and trim beginning
                print("  Using fallback pitch processing (over-stretch + trim)...")
                pitch_corrected_path = audio_processor.prepare_fallback_pitch_sample(
                    sample_to_process,
                    output_path=str(TEMP_DIR / "pitch_corrected.wav"),
                    freq=293.66,  # D4
                    stretch_rate=0.2  # 5x longer with RubberBand crisp6
                )
            else:
                # Normal pitch correction
                pitch_corrected_path = audio_processor.pitch_correct_sample(
                    sample_to_process,
                    output_path=str(TEMP_DIR / "pitch_corrected.wav"),
                    freq=293.66  # D4
                )
            
            if pitch_corrected_path:
                print(f"Pitch corrected sample ready: {pitch_corrected_path}")
                arranger.pitch_sample_path = pitch_corrected_path
                arranger.pitch_sample_src_info = best_pitch_sample
            else:
                print("Pitch correction failed, continuing without pitch instrument")
        else:
            print("No suitable pitch sample found, continuing without pitch instrument")
        
        # Step 2d: Find and pitch-correct MELODY sample (second pitch sample for top-right)
        print("\nFinding best melody sample (for G Minor melodies)...")
        # Exclude the samples already used
        used_for_melody = [s['path'] for s in pure_vowels]
        if best_pitch_sample:
            used_for_melody.append(best_pitch_sample['path'])
        
        best_melody_sample, is_melody_fallback = audio_processor.get_best_pitch_sample(
            samples, exclude_paths=used_for_melody, random_pick=True
        )
        melody_corrected_path = None
        if best_melody_sample:
            # Log transient quality
            melody_transient = best_melody_sample.get('transient_score', 0.5)
            melody_trim = best_melody_sample.get('trim_samples', 0)
            print(f"Best melody sample: {Path(best_melody_sample['path']).name}")
            print(f"  Transient score: {melody_transient:.2f} (1.0=sharp, 0.0=slow)")
            
            # First, trim slow attack if needed
            melody_to_process = best_melody_sample['path']
            if melody_trim > 0 and melody_transient < 0.7:
                melody_trimmed_path = str(TEMP_DIR / "melody_trimmed.wav")
                melody_to_process = audio_processor.trim_sample_attack(
                    best_melody_sample['path'], melody_trimmed_path, melody_trim
                )
            
            if is_melody_fallback:
                # Use fallback processing: over-stretch and trim beginning
                print("  Using fallback melody processing (over-stretch + trim)...")
                melody_corrected_path = audio_processor.prepare_fallback_pitch_sample(
                    melody_to_process,
                    output_path=str(TEMP_DIR / "melody_corrected.wav"),
                    freq=293.66,  # D4
                    stretch_rate=0.2  # 5x longer with RubberBand crisp6
                )
            else:
                # Normal pitch correction
                melody_corrected_path = audio_processor.pitch_correct_sample(
                    melody_to_process,
                    output_path=str(TEMP_DIR / "melody_corrected.wav"),
                    freq=293.66  # D4
                )
            
            if melody_corrected_path:
                print(f"Melody corrected sample ready: {melody_corrected_path}")
                arranger.melody_sample_path = melody_corrected_path
                arranger.melody_sample_src_info = best_melody_sample
            else:
                print("Melody correction failed, continuing without melody instrument")
        else:
            print("No suitable melody sample found, continuing without melody instrument")
        
        # Step 2e: Find and pitch-correct bass sample to D2 (73.42 Hz)
        print("\nFinding best bass instrument sample...")
        # Exclude pitch, melody, and vocal samples from bass selection
        used_paths = [s['path'] for s in pure_vowels]
        if pitch_corrected_path and best_pitch_sample:
            used_paths.append(best_pitch_sample['path'])
        if melody_corrected_path and best_melody_sample:
            used_paths.append(best_melody_sample['path'])
        
        best_bass_sample, is_bass_fallback = audio_processor.get_best_bass_sample(
            samples, exclude_paths=used_paths, random_pick=True
        )
        bass_corrected_path = None
        if best_bass_sample:
            print(f"Best bass sample: {best_bass_sample['path']}")
            
            # Pitch correct to D2 (73.42 Hz)
            bass_corrected_path = audio_processor.prepare_bass_sample(
                best_bass_sample['path'],
                output_path=str(TEMP_DIR / "bass_corrected.wav"),
                freq=73.42  # D2
            )
            
            if bass_corrected_path:
                print(f"Bass sample ready (D2): {bass_corrected_path}")
                arranger.bass_sample_path = bass_corrected_path
                arranger.bass_sample_src_info = best_bass_sample
            else:
                print("Bass preparation failed, continuing without bass instrument")
        else:
            print("No suitable bass sample found, continuing without bass instrument")
        
        # Step 2f: Find and prepare chord/pad sample (for chorus pads)
        print("\nFinding best chord sample (for chorus pads / Zorammi chords)...")
        # Exclude samples already used for other instruments
        used_for_chord = [s['path'] for s in pure_vowels]
        if best_pitch_sample:
            used_for_chord.append(best_pitch_sample['path'])
        if best_melody_sample:
            used_for_chord.append(best_melody_sample['path'])
        if best_bass_sample:
            used_for_chord.append(best_bass_sample['path'])
        
        best_chord_sample, is_chord_fallback = audio_processor.get_best_zorammi_chord_sample(
            samples,
            exclude_paths=used_for_chord,
            random_pick=True,
            min_duration_sec=0.25  # 25 centiseconds minimum
        )
        chord_prepared_path = None
        zorammi_chords_path = None
        if best_chord_sample:
            print(f"Best chord sample: {Path(best_chord_sample['path']).name}")
            
            # Prepare chord sample with time-stretch (0.1 = 10x longer) and lowpass at 2900Hz
            chord_prepared_path = audio_processor.prepare_chord_sample(
                best_chord_sample['path'],
                output_path=str(TEMP_DIR / "chord_sample.wav"),
                freq=293.66,  # D4
                stretch_rate=0.1,  # Very slow/stretched for pad sound
                lowpass_freq=2900  # Gentle lowpass to soften
            )
            
            if chord_prepared_path:
                print(f"Chord sample ready: {chord_prepared_path}")
                arranger.chord_sample_path = chord_prepared_path
                arranger.chord_sample_src_info = best_chord_sample
            else:
                print("Chord preparation failed, continuing without chord pads")

            # Prepare Zorammi chords variant (PitchCorrector297 + RubberBand crisp6 + trim)
            zorammi_chords_path = audio_processor.prepare_zorammi_chords_sample(
                best_chord_sample['path'],
                output_path=str(TEMP_DIR / "zorammi_chords.wav"),
                freq=293.66,
                stretch_rate=0.15,  # ~6.7x longer with RubberBand crisp6
            )
            if zorammi_chords_path:
                print(f"Zorammi chords sample ready: {zorammi_chords_path}")
                arranger.zorammi_chords_sample_path = zorammi_chords_path
                arranger.zorammi_chords_src_info = best_chord_sample
            else:
                print("Zorammi chords preparation failed")
        else:
            print("No suitable chord sample found, continuing without chord pads/Zorammi chords")
        
        # Step 2g: Find and prepare AWESOMENESS sample (for last chorus, center-right position)
        print("\nFinding best awesomeness sample (for last chorus)...")
        # Exclude samples already used for other instruments
        used_for_awesomeness = [s['path'] for s in pure_vowels]
        if best_pitch_sample:
            used_for_awesomeness.append(best_pitch_sample['path'])
        if best_melody_sample:
            used_for_awesomeness.append(best_melody_sample['path'])
        if best_bass_sample:
            used_for_awesomeness.append(best_bass_sample['path'])
        if best_chord_sample:
            used_for_awesomeness.append(best_chord_sample['path'])
        
        best_awesomeness_sample, is_awesomeness_fallback = audio_processor.get_best_pitch_sample(
            samples, exclude_paths=used_for_awesomeness, random_pick=True
        )
        awesomeness_corrected_path = None
        if best_awesomeness_sample:
            awesomeness_transient = best_awesomeness_sample.get('transient_score', 0.5)
            awesomeness_trim = best_awesomeness_sample.get('trim_samples', 0)
            print(f"Best awesomeness sample: {Path(best_awesomeness_sample['path']).name}")
            print(f"  Transient score: {awesomeness_transient:.2f}")
            
            # Trim slow attack if needed
            awesomeness_to_process = best_awesomeness_sample['path']
            if awesomeness_trim > 0 and awesomeness_transient < 0.7:
                awesomeness_trimmed_path = str(TEMP_DIR / "awesomeness_trimmed.wav")
                awesomeness_to_process = audio_processor.trim_sample_attack(
                    best_awesomeness_sample['path'], awesomeness_trimmed_path, awesomeness_trim
                )
            
            # Awesomeness sample needs stretching (like pad but less extreme)
            # stretch_rate=0.3 means ~3.3x longer
            awesomeness_corrected_path = audio_processor.prepare_awesomeness_sample(
                awesomeness_to_process,
                output_path=str(TEMP_DIR / "awesomeness_corrected.wav"),
                freq=293.66,  # D4
                stretch_rate=0.3  # ~3.3x longer (less than pad's 0.1 which is 10x)
            )
            
            if awesomeness_corrected_path:
                print(f"Awesomeness sample ready: {awesomeness_corrected_path}")
                arranger.awesomeness_sample_path = awesomeness_corrected_path
                arranger.awesomeness_sample_src_info = best_awesomeness_sample
            else:
                print("Awesomeness preparation failed, continuing without awesomeness")
        else:
            print("No suitable awesomeness sample found, continuing without awesomeness")
        
        # Step 2h: Find and prepare EPICNESS PITCH sample (for epicness section, top-left position)
        print("\nFinding best epicness pitch sample (for epicness section)...")
        # Exclude samples already used for other instruments
        used_for_epicness_pitch = [s['path'] for s in pure_vowels]
        if best_pitch_sample:
            used_for_epicness_pitch.append(best_pitch_sample['path'])
        if best_melody_sample:
            used_for_epicness_pitch.append(best_melody_sample['path'])
        if best_bass_sample:
            used_for_epicness_pitch.append(best_bass_sample['path'])
        if best_chord_sample:
            used_for_epicness_pitch.append(best_chord_sample['path'])
        if best_awesomeness_sample:
            used_for_epicness_pitch.append(best_awesomeness_sample['path'])
        
        best_epicness_pitch_sample, is_epicness_pitch_fallback = audio_processor.get_best_pitch_sample(
            samples, exclude_paths=used_for_epicness_pitch, random_pick=True
        )
        epicness_pitch_corrected_path = None
        if best_epicness_pitch_sample:
            epicness_pitch_transient = best_epicness_pitch_sample.get('transient_score', 0.5)
            epicness_pitch_trim = best_epicness_pitch_sample.get('trim_samples', 0)
            print(f"Best epicness pitch sample: {Path(best_epicness_pitch_sample['path']).name}")
            print(f"  Transient score: {epicness_pitch_transient:.2f}")
            
            # Trim slow attack if needed
            epicness_pitch_to_process = best_epicness_pitch_sample['path']
            if epicness_pitch_trim > 0 and epicness_pitch_transient < 0.7:
                epicness_pitch_trimmed_path = str(TEMP_DIR / "epicness_pitch_trimmed.wav")
                epicness_pitch_to_process = audio_processor.trim_sample_attack(
                    best_epicness_pitch_sample['path'], epicness_pitch_trimmed_path, epicness_pitch_trim
                )
            
            if is_epicness_pitch_fallback:
                print("  Using fallback epicness pitch processing (over-stretch + trim)...")
                epicness_pitch_corrected_path = audio_processor.prepare_fallback_pitch_sample(
                    epicness_pitch_to_process,
                    output_path=str(TEMP_DIR / "epicness_pitch_corrected.wav"),
                    freq=293.66,  # D4
                    stretch_rate=0.2  # 5x longer with RubberBand crisp6
                )
            else:
                epicness_pitch_corrected_path = audio_processor.pitch_correct_sample(
                    epicness_pitch_to_process,
                    output_path=str(TEMP_DIR / "epicness_pitch_corrected.wav"),
                    freq=293.66  # D4
                )
            
            if epicness_pitch_corrected_path:
                print(f"Epicness pitch sample ready: {epicness_pitch_corrected_path}")
                arranger.epicness_pitch_sample_path = epicness_pitch_corrected_path
                arranger.epicness_pitch_sample_src_info = best_epicness_pitch_sample
            else:
                print("Epicness pitch preparation failed, continuing without epicness pitch")
        else:
            print("No suitable epicness pitch sample found, continuing without epicness pitch")
        
        # Step 2i: Get percussion samples (always use something, even if not ideal)
        print("\nFinding percussion samples...")
        percussion = audio_processor.get_best_percussion_samples(samples)
        if percussion['kick']:
            arranger.kick_sample = percussion['kick']
            print(f"  Kick: {Path(percussion['kick']['path']).name}")
        if percussion['snare']:
            arranger.snare_sample = percussion['snare']
            print(f"  Snare: {Path(percussion['snare']['path']).name}")
        if percussion['hihat']:
            arranger.hihat_sample = percussion['hihat']
            print(f"  Hihat: {Path(percussion['hihat']['path']).name}")
        
        # Step 3: Generate song structure
        report("🎼 Generating song structure...", 35)
        song_structure = pattern_generator.create_song_structure(samples, remix_length=remix_length)
        
        # Print song structure
        print("\nSong structure:")
        for section in song_structure:
            print(f"- {section['name'].title()}: {section['length_bars']} bars")
        
        # Step 4: Render each section
        report("🎹 Placing audio samples...", 45)
        total_sections = max(1, len(song_structure))
        for idx, section in enumerate(song_structure, start=1):
            pct = 45 + int(35 * (idx / total_sections))
            report(f"🎹 Rendering {section['name']}...", pct)
            arranger.render_section(section, samples)
        
        # Step 5: Mixdown and export
        report("🎚️ Mixing down audio...", 85)
        # Optional base bed path overlay at -5 dB
        base_bed = RESOURCE_DIR / "base.wav"
        final_audio = arranger.mixdown(audio_output_path, base_bed_path=str(base_bed) if base_bed.exists() else None)
        # Step 6: Render aligned video
        report("🎬 Rendering video...", 92)
        video_output = arranger.render_video(
            source_video_path=video_path,
            audio_path=final_audio,
            output_path=video_output_path,
            resolution=(854, 480),
            progress_callback=progress_callback,
        )

        report("✅ Finalizing...", 99)
        print(f"\n🎵 Sparta Remix generated successfully! 🎵")
        print(f"Output file: {final_audio}")
        if video_output:
            print(f"Video file: {video_output}")
            report("✅ Done!", 100)
            return video_output
        report("✅ Done!", 100)
        return final_audio
        
    except Exception as e:
        print(f"\n❌ Error generating Sparta Remix: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a Sparta Remix from a video file.')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('-o', '--output', type=str, help='Output file path (default: output/sparta_remix.wav)')
    parser.add_argument('--bpm', type=int, default=140, help='Tempo in BPM (default: 140)')
    parser.add_argument('--length', type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                        help='Remix length: 1=Intro, 2=Intro+Chorus, 3=Half, 4=Extended, 5=Almost Full, 6=Full (default: 6)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the video
    output_path = process_video(
        video_path=args.input_video,
        output_path=args.output,
        bpm=args.bpm,
        remix_length=args.length
    )
    
    if output_path:
        print(f"\n✨ All done! Your Sparta Remix is ready at: {output_path}")
    else:
        print("\n❌ Failed to generate Sparta Remix. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
