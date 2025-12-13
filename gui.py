import threading
import sys
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import os
import winsound
import customtkinter as ctk

from main import process_video
from config import OUTPUT_DIR, RESOURCE_DIR, BASE_DIR

_LOG_PATH = None
_LOG_FH = None
if getattr(sys, "frozen", False):
    try:
        _LOG_PATH = BASE_DIR / "sparta_gui.log"
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
        sys.stdout = _LOG_FH
        sys.stderr = _LOG_FH
    except Exception:
        _LOG_PATH = None

# Fun SFX for Zorammi toggle
GUN_SFX_PATH = RESOURCE_DIR / "build" / "gun" / "GUN.wav"

# Set customtkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class ModernSpartaGUI(ctk.CTk):
    """Modern GUI for Sparta Remix Generator with video preview and quote selection."""

    # Color palette from user mockup
    ASH_GREY = "#BAC1B8"
    PACIFIC_BLUE = "#58A4B0"
    TURF_GREEN = "#0C7C59"
    JET_BLACK = "#2B303A"
    JET_BLACK_LIGHT = "#3d4452"  # Lighter shade for gradient effect
    BURNT_TANGERINE = "#D64933"
    
    # Derived colors
    BG_DARK = JET_BLACK
    BG_CARD = "#353b47"  # Slightly lighter than jet black
    FG_TEXT = "#FFFFFF"
    FG_DIM = ASH_GREY
    ACCENT = BURNT_TANGERINE
    ACCENT_HOVER = "#e05a45"  # Lighter tangerine
    SUCCESS = TURF_GREEN
    INFO = PACIFIC_BLUE
    
    def __init__(self):
        super().__init__()
        self.title("Sparta Remix Generator")
        self.geometry("1000x520")
        self.configure(fg_color=self.BG_DARK)
        self.resizable(True, True)
        self.minsize(980, 500)

        # Variables
        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.bpm = tk.StringVar(value="140")
        self.remix_length = tk.IntVar(value=6)  # 1-6 scale for remix length
        self.status_text = tk.StringVar(value="Ready to remix!")
        self.progress_text = tk.StringVar(value="0%")
        
        # Instrument toggles (all enabled by default)
        self.enable_kick = tk.BooleanVar(value=True)
        self.enable_snare = tk.BooleanVar(value=True)
        self.enable_hihat = tk.BooleanVar(value=True)
        self.enable_bass = tk.BooleanVar(value=True)
        self.enable_pitch = tk.BooleanVar(value=True)
        self.enable_melody = tk.BooleanVar(value=True)
        self.enable_chords = tk.BooleanVar(value=True)
        self.enable_vocals = tk.BooleanVar(value=True)
        self.enable_awesomeness = tk.BooleanVar(value=True)
        self.enable_epicness_pitch = tk.BooleanVar(value=True)
        self.enable_zorammi_style = tk.BooleanVar(value=False)
        # Map vars for easier bulk updates
        self.toggle_vars = {
            'kick': self.enable_kick,
            'snare': self.enable_snare,
            'hihat': self.enable_hihat,
            'bass': self.enable_bass,
            'pitch': self.enable_pitch,
            'melody': self.enable_melody,
            'chords': self.enable_chords,
            'vocals': self.enable_vocals,
            'awesomeness': self.enable_awesomeness,
            'epicness_pitch': self.enable_epicness_pitch,
            'zorammi_style': self.enable_zorammi_style,
        }
        self.toggle_buttons = {}
        self._pre_zorammi_states = {}
        
        # Quote selection (start/end in seconds)
        self.quote_start = tk.DoubleVar(value=0.0)
        self.quote_end = tk.DoubleVar(value=0.0)
        self.video_duration = 0.0
        
        # Video capture for preview
        self.cap = None
        self.preview_image = None
        self.output_preview_image = None
        
        # Render control
        self.render_thread = None
        self.terminate_flag = False
        
        self._build_ui()
        
        # Bind video path change to update preview
        self.video_path.trace_add("write", self._on_video_change)

    def _shake_window(self, cycles: int = 14, offset: int = 18, interval_ms: int = 22):
        """Simple window shake effect (longer + more extreme)."""
        try:
            geo = self.geometry()
            parts = geo.split("+")
            if len(parts) < 3:
                return
            base_x, base_y = int(parts[1]), int(parts[2])

            def do_shake(i=0):
                if i >= cycles:
                    self.geometry(f"+{base_x}+{base_y}")
                    return
                dx = offset if i % 2 == 0 else -offset
                dy = (-offset // 2) if (i % 4 == 0) else (offset // 2)
                self.geometry(f"+{base_x + dx}+{base_y + dy}")
                self.after(interval_ms, lambda: do_shake(i + 1))

            do_shake()
        except Exception as e:
            print(f"Shake failed: {e}")

    def _on_zorammi_toggle(self):
        """Handle Zorammi Style toggle with sound + delayed shake."""
        if self.enable_zorammi_style.get():
            try:
                if GUN_SFX_PATH.exists():
                    winsound.PlaySound(str(GUN_SFX_PATH), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print(f"Zorammi SFX failed: {e}")
            self.after(350, self._apply_zorammi_shake)
        self._apply_zorammi_ui_state()

    def _apply_zorammi_shake(self):
        if self.enable_zorammi_style.get():
            self._shake_window()

    def _build_ui(self):
        """Build the main UI layout matching the user's mockup."""
        # Main container - compact padding
        main_frame = ctk.CTkFrame(self, fg_color=self.BG_DARK, corner_radius=0)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)
        
        # === TOP ROW: Title + Credit Box (side by side) ===
        top_row = ctk.CTkFrame(main_frame, fg_color="transparent")
        top_row.pack(fill=tk.X, pady=(0, 6))
        
        # Title on left
        title_label = ctk.CTkLabel(
            top_row, 
            text="Sparta Remix Generator",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=self.FG_TEXT
        )
        title_label.pack(side=tk.LEFT)
        
        # Credit box next to title (Pacific Blue background)
        credit_frame = ctk.CTkFrame(top_row, fg_color=self.PACIFIC_BLUE, corner_radius=5)
        credit_frame.pack(side=tk.LEFT, padx=(15, 0))
        
        credit_text = ctk.CTkLabel(
            credit_frame,
            text="If used and published on any social media platforms,\nplease do not hesitate to credit me via my YouTube\nchannel @krasen671 or the link to my Github page.",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color="#FFFFFF",
            justify="left"
        )
        credit_text.pack(padx=10, pady=5)
        
        # === INPUT VIDEO ROW ===
        input_row = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_row.pack(fill=tk.X, pady=(0, 6))
        
        input_label = ctk.CTkLabel(
            input_row, 
            text="Input Video:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=self.FG_TEXT
        )
        input_label.pack(side=tk.LEFT)
        
        self.input_entry = ctk.CTkEntry(
            input_row, 
            textvariable=self.video_path,
            width=270,
            height=26,
            fg_color=self.BG_CARD,
            border_color=self.ASH_GREY,
            text_color=self.FG_DIM
        )
        self.input_entry.pack(side=tk.LEFT, padx=(8, 8))
        
        browse_btn = ctk.CTkButton(
            input_row,
            text="Browse",
            width=65,
            height=26,
            fg_color=self.PACIFIC_BLUE,
            hover_color=self.TURF_GREEN,
            font=ctk.CTkFont(size=11),
            command=self._browse_video
        )
        browse_btn.pack(side=tk.LEFT)
        
        # === MAIN CONTENT AREA (3 columns using grid) ===
        content_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        content_frame.pack(fill=tk.X, expand=False, pady=(0, 0), anchor="nw")

        # Grid layout for 3 columns
        content_frame.grid_columnconfigure(0, weight=0, minsize=370)  # Left (fixed-ish)
        content_frame.grid_columnconfigure(1, weight=0, minsize=180)  # Middle (fixed-ish)
        content_frame.grid_columnconfigure(2, weight=0, minsize=450)  # Right (fixed-ish)
        content_frame.grid_rowconfigure(0, weight=0)
        
        # === LEFT COLUMN: Input Preview + Selection + Remix Length ===
        left_col = ctk.CTkFrame(content_frame, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nw", padx=(0, 10))

        preview_w = 350
        preview_h = 180
        slider_w = 220

        # Input video preview (fixed size like mockup)
        self.preview_canvas = tk.Canvas(
            left_col,
            width=preview_w,
            height=preview_h,
            bg=self.JET_BLACK,
            highlightthickness=2,
            highlightbackground=self.ASH_GREY
        )
        self.preview_canvas.pack(anchor="w", pady=(0, 6))
        self.preview_canvas.create_text(
            preview_w // 2,
            preview_h // 2,
            text="(preview of video)",
            fill=self.ASH_GREY,
            font=("Segoe UI", 10)
        )
        
        # SELECTION section
        selection_label = ctk.CTkLabel(
            left_col,
            text="SELECTION:",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.FG_TEXT
        )
        selection_label.pack(anchor="w", pady=(0, 3))
        
        # Quote Start slider
        start_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        start_frame.pack(fill=tk.X, pady=(0, 2))
        
        ctk.CTkLabel(
            start_frame,
            text="Quote Start:",
            font=ctk.CTkFont(size=10),
            text_color=self.FG_DIM,
            width=65
        ).pack(side=tk.LEFT)
        
        self.start_slider = ctk.CTkSlider(
            start_frame,
            from_=0,
            to=100,
            variable=self.quote_start,
            height=14,
            width=slider_w,
            fg_color=self.BG_CARD,
            progress_color=self.PACIFIC_BLUE,
            button_color=self.PACIFIC_BLUE,
            button_hover_color=self.TURF_GREEN,
            command=self._on_start_change
        )
        self.start_slider.pack(side=tk.LEFT, padx=(4, 6))
        
        self.start_label = ctk.CTkLabel(
            start_frame,
            text="0.00s",
            font=ctk.CTkFont(size=10),
            text_color=self.FG_DIM,
            width=45
        )
        self.start_label.pack(side=tk.LEFT)
        
        # Quote End slider
        end_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        end_frame.pack(fill=tk.X, pady=(0, 2))
        
        ctk.CTkLabel(
            end_frame,
            text="Quote End:",
            font=ctk.CTkFont(size=10),
            text_color=self.FG_DIM,
            width=65
        ).pack(side=tk.LEFT)
        
        self.end_slider = ctk.CTkSlider(
            end_frame,
            from_=0,
            to=100,
            variable=self.quote_end,
            height=14,
            width=slider_w,
            fg_color=self.BG_CARD,
            progress_color=self.PACIFIC_BLUE,
            button_color=self.PACIFIC_BLUE,
            button_hover_color=self.TURF_GREEN,
            command=self._on_end_change
        )
        self.end_slider.pack(side=tk.LEFT, padx=(4, 6))
        
        self.end_label = ctk.CTkLabel(
            end_frame,
            text="0.00s",
            font=ctk.CTkFont(size=10),
            text_color=self.FG_DIM,
            width=45
        )
        self.end_label.pack(side=tk.LEFT)
        
        # Duration info
        self.duration_label = ctk.CTkLabel(
            left_col,
            text="Selected: 0.00s (from 0.00s to 0.00s)",
            font=ctk.CTkFont(size=9),
            text_color=self.PACIFIC_BLUE
        )
        self.duration_label.pack(anchor="w", pady=(2, 5))
        
        # Remix Length slider (IN LEFT COLUMN)
        length_frame = ctk.CTkFrame(left_col, fg_color=self.PACIFIC_BLUE, corner_radius=5)
        length_frame.pack(fill=tk.X, pady=(0, 0))
        
        length_inner = ctk.CTkFrame(length_frame, fg_color="transparent")
        length_inner.pack(fill=tk.X, padx=8, pady=5)
        
        ctk.CTkLabel(
            length_inner,
            text="Remix Length:",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color="#FFFFFF"
        ).pack(side=tk.LEFT)
        
        self.length_slider = ctk.CTkSlider(
            length_inner,
            from_=1,
            to=6,
            number_of_steps=5,
            variable=self.remix_length,
            height=14,
            width=slider_w,
            fg_color=self.JET_BLACK,
            progress_color=self.TURF_GREEN,
            button_color="#FFFFFF",
            button_hover_color=self.ASH_GREY,
            command=self._on_length_change
        )
        self.length_slider.pack(side=tk.LEFT, padx=(8, 8))
        
        self.length_label = ctk.CTkLabel(
            length_inner,
            text="6 - Full Remix (32 bars)",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color="#FFFFFF"
        )
        self.length_label.pack(side=tk.LEFT)
        
        # === MIDDLE COLUMN: Sample Toggles ===
        middle_col = ctk.CTkFrame(content_frame, fg_color="transparent")
        middle_col.grid(row=0, column=1, sticky="nw", padx=8, pady=(0, 0))
        
        toggles_label = ctk.CTkLabel(
            middle_col,
            text="SAMPLE TOGGLES:",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.FG_TEXT
        )
        toggles_label.pack(anchor="w", pady=(0, 4))
        
        # Toggle checkboxes
        toggle_items = [
            ("KICK", self.enable_kick, 'kick'),
            ("SNARE", self.enable_snare, 'snare'),
            ("HIHAT", self.enable_hihat, 'hihat'),
            ("BASS", self.enable_bass, 'bass'),
            ("VOCALS", self.enable_vocals, 'vocals'),
            ("PITCH", self.enable_pitch, 'pitch'),
            ("MELODY", self.enable_melody, 'melody'),
            ("CHORDS/PADS", self.enable_chords, 'chords'),
            ("AWESOMENESS", self.enable_awesomeness, 'awesomeness'),
            ("EPICNESS PITCH", self.enable_epicness_pitch, 'epicness_pitch'),
        ]
        
        for text, var, key in toggle_items:
            cb = ctk.CTkCheckBox(
                middle_col,
                text=text,
                variable=var,
                font=ctk.CTkFont(family="Segoe UI", size=10),
                text_color=self.FG_TEXT,
                fg_color=self.TURF_GREEN,
                hover_color=self.PACIFIC_BLUE,
                border_color=self.ASH_GREY,
                checkmark_color=self.FG_TEXT,
                checkbox_width=16,
                checkbox_height=16
            )
            cb.pack(anchor="w", pady=1)
            self.toggle_buttons[key] = cb
        
        # Zorammi Style toggle (special)
        zorammi_cb = ctk.CTkCheckBox(
            middle_col,
            text="ZORAMMI STYLE",
            variable=self.enable_zorammi_style,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.BURNT_TANGERINE,
            fg_color=self.BURNT_TANGERINE,
            hover_color=self.ACCENT_HOVER,
            border_color=self.BURNT_TANGERINE,
            checkmark_color=self.FG_TEXT,
            checkbox_width=16,
            checkbox_height=16,
            command=self._on_zorammi_toggle
        )
        zorammi_cb.pack(anchor="w", pady=(5, 0))
        self.toggle_buttons['zorammi_style'] = zorammi_cb
        
        # === RIGHT COLUMN: Output Preview + Generate Button ===
        right_col = ctk.CTkFrame(content_frame, fg_color="transparent")
        right_col.grid(row=0, column=2, sticky="nw", padx=(10, 0))

        out_w = 440
        out_entry_w = 350
        
        # Output path label
        output_label = ctk.CTkLabel(
            right_col,
            text="RESULT/OUTPUT VIDEO:",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.FG_TEXT
        )
        output_label.pack(anchor="w", pady=(0, 3))
        
        # Output path entry
        output_path_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        output_path_frame.pack(fill=tk.X, pady=(0, 5), anchor="w")
        
        self.output_entry = ctk.CTkEntry(
            output_path_frame,
            textvariable=self.output_path,
            width=out_entry_w,
            height=24,
            fg_color=self.BG_CARD,
            border_color=self.ASH_GREY,
            text_color=self.FG_DIM
        )
        self.output_entry.pack(side=tk.LEFT)
        
        output_browse_btn = ctk.CTkButton(
            output_path_frame,
            text="...",
            width=30,
            height=24,
            fg_color=self.PACIFIC_BLUE,
            hover_color=self.TURF_GREEN,
            command=self._browse_output
        )
        output_browse_btn.pack(side=tk.LEFT, padx=(4, 0))
        
        # Output preview canvas (fixed size like mockup)
        self.output_canvas = tk.Canvas(
            right_col,
            width=out_w,
            height=preview_h,
            bg=self.JET_BLACK,
            highlightthickness=2,
            highlightbackground=self.ASH_GREY
        )
        self.output_canvas.pack(anchor="w", pady=(0, 6))
        self.output_canvas.create_text(
            out_w // 2,
            preview_h // 2,
            text="(PREVIEW OF OUTPUT)",
            fill=self.ASH_GREY,
            font=("Segoe UI", 10)
        )
        
        # Play button row
        play_row = ctk.CTkFrame(right_col, fg_color="transparent", width=out_w)
        play_row.pack(anchor="w", pady=(6, 8))
        
        self.play_button = ctk.CTkButton(
            play_row,
            text="▶",
            width=36,
            height=32,
            fg_color=self.BG_CARD,
            hover_color=self.PACIFIC_BLUE,
            border_width=1,
            border_color=self.ASH_GREY,
            command=self._play_output
        )
        self.play_button.pack(side=tk.LEFT)
        
        # Generate button - FULL WIDTH, SEPARATE ROW
        self.run_button = ctk.CTkButton(
            right_col,
            text="🎵 Generate Sparta Remix",
            height=42,
            width=out_w,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=self.BURNT_TANGERINE,
            hover_color=self.ACCENT_HOVER,
            command=self._run_async
        )
        self.run_button.pack(anchor="w", pady=(0, 0))
        
        # Terminate button (hidden initially, will be shown during render)
        self.terminate_button = ctk.CTkButton(
            right_col,
            text="⏹ Stop Render",
            height=36,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=self.BG_CARD,
            hover_color="#5a3030",
            border_width=1,
            border_color=self.BURNT_TANGERINE,
            command=self._terminate_render
        )
        # Hidden by default
        
        # === BOTTOM: Status label ===
        status_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        status_frame.pack(fill=tk.X, pady=(6, 0))

        self.progress_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.progress_text,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=self.PACIFIC_BLUE
        )
        self.progress_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_text,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=self.TURF_GREEN
        )
        self.status_label.pack(side=tk.LEFT)
        
    def _play_output(self):
        """Open the output video file if it exists."""
        output = self.output_path.get().strip()
        if output and Path(output).exists():
            os.startfile(output)
        else:
            messagebox.showinfo("No Output", "No output video to play yet. Generate a remix first!")

    def _apply_zorammi_ui_state(self):
        """Grey/disable controls based on Zorammi style rules."""
        zorammi_on = self.enable_zorammi_style.get()

        if zorammi_on and not self._pre_zorammi_states:
            for name, var in self.toggle_vars.items():
                self._pre_zorammi_states[name] = var.get()

        allowed_on = {'kick', 'snare', 'hihat', 'bass', 'vocals'}
        for name, btn in self.toggle_buttons.items():
            if name == 'zorammi_style':
                continue
            is_allowed = (name in allowed_on)
            if zorammi_on:
                self.toggle_vars[name].set(is_allowed)
                if is_allowed:
                    btn.configure(state="normal")
                else:
                    btn.configure(state="disabled")
            else:
                if self._pre_zorammi_states:
                    self.toggle_vars[name].set(self._pre_zorammi_states.get(name, self.toggle_vars[name].get()))
                btn.configure(state="normal")

        if zorammi_on:
            for key in ['chords', 'melody', 'pitch', 'awesomeness', 'epicness_pitch']:
                if key in self.toggle_buttons:
                    self.toggle_vars[key].set(False)
                    self.toggle_buttons[key].configure(state="disabled")
        else:
            self._pre_zorammi_states = {}

    def _browse_video(self):
        """Open file dialog for input video."""
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            # Auto-set output to same directory
            video_dir = Path(path).parent
            video_name = Path(path).stem
            self.output_path.set(str(video_dir / f"{video_name}_sparta_remix.mp4"))

    def _browse_output(self):
        """Open file dialog for output."""
        initial_dir = ""
        if self.video_path.get():
            initial_dir = str(Path(self.video_path.get()).parent)
        
        path = filedialog.asksaveasfilename(
            title="Save output",
            initialdir=initial_dir,
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("WAV Audio", "*.wav")]
        )
        if path:
            self.output_path.set(path)

    def _on_video_change(self, *args):
        """Called when video path changes - load preview."""
        path = self.video_path.get()
        if path and Path(path).exists():
            self._load_video_preview(path)

    def _load_video_preview(self, path: str):
        """Load video and show first frame as preview."""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                return
            
            # Get video info
            self.video_duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
            
            # Update sliders
            self.start_slider.configure(to=self.video_duration)
            self.end_slider.configure(to=self.video_duration)
            self.quote_start.set(0)
            self.quote_end.set(self.video_duration)
            self._update_time_labels()
            
            # Show first frame
            self._show_frame_at(0)
            
        except Exception as e:
            print(f"Error loading video preview: {e}")

    def _show_frame_at(self, time_sec: float):
        """Show video frame at specified time."""
        if not self.cap:
            return
        
        try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(time_sec * fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to fit the fixed canvas size (avoid stretched-strip layouts)
                self.preview_canvas.update_idletasks()
                configured_w = int(float(self.preview_canvas.cget("width") or 0))
                configured_h = int(float(self.preview_canvas.cget("height") or 0))
                canvas_w = self.preview_canvas.winfo_width() or configured_w
                canvas_h = self.preview_canvas.winfo_height() or configured_h
                if canvas_w <= 1:
                    canvas_w = configured_w
                if canvas_h <= 1:
                    canvas_h = configured_h
                h, w = frame.shape[:2]
                scale = min(canvas_w/w, canvas_h/h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Convert to PhotoImage
                img = Image.fromarray(frame)
                self.preview_image = ImageTk.PhotoImage(img)
                
                # Clear and draw
                self.preview_canvas.delete("all")
                x_offset = (canvas_w - new_w) // 2
                y_offset = (canvas_h - new_h) // 2
                self.preview_canvas.create_image(x_offset, y_offset, 
                                                 anchor=tk.NW, 
                                                 image=self.preview_image)
                
                # Draw time indicator
                time_str = f"{time_sec:.2f}s"
                self.preview_canvas.create_text(8, 8, 
                                               text=time_str,
                                               fill=self.BURNT_TANGERINE,
                                               font=("Segoe UI", 9, "bold"),
                                               anchor=tk.NW)
        except Exception as e:
            print(f"Error showing frame: {e}")

    def _on_length_change(self, value):
        """Handle remix length slider change."""
        length = int(round(float(value)))
        self.remix_length.set(length)
        
        # Update label with description
        length_info = {
            1: "1 - Intro Only (2 bars)",
            2: "2 - Intro + Chorus (6 bars)",
            3: "3 - Half Remix (12 bars)",
            4: "4 - Extended (20 bars)",
            5: "5 - Almost Full (24 bars)",
            6: "6 - Full Remix (32 bars)",
        }
        self.length_label.configure(text=length_info.get(length, f"{length} - Custom"))

    def _on_start_change(self, value):
        """Handle start slider change."""
        start = float(value)
        # Ensure start <= end
        if start > self.quote_end.get():
            self.quote_end.set(start)
        self._update_time_labels()
        self._show_frame_at(start)

    def _on_end_change(self, value):
        """Handle end slider change."""
        end = float(value)
        # Ensure end >= start
        if end < self.quote_start.get():
            self.quote_start.set(end)
        self._update_time_labels()
        self._show_frame_at(end)

    def _update_time_labels(self):
        """Update time display labels."""
        start = self.quote_start.get()
        end = self.quote_end.get()
        duration = end - start
        
        self.start_label.configure(text=f"{start:.2f}s")
        self.end_label.configure(text=f"{end:.2f}s")
        self.duration_label.configure(text=f"Selected: {duration:.2f}s (from {start:.2f}s to {end:.2f}s)")

    def _run_async(self):
        """Run generation on background thread."""
        video = self.video_path.get().strip()
        if not video:
            messagebox.showwarning("Missing file", "Please choose an input video.")
            return
        if not Path(video).exists():
            messagebox.showerror("File not found", f"Can't find video: {video}")
            return

        try:
            bpm_val = int(self.bpm.get())
        except ValueError:
            messagebox.showerror("Invalid BPM", "BPM must be a number.")
            return

        output = self.output_path.get().strip()
        if not output:
            video_dir = Path(video).parent
            video_name = Path(video).stem
            output = str(video_dir / f"{video_name}_sparta_remix.mp4")

        # Get quote range
        quote_start = self.quote_start.get()
        quote_end = self.quote_end.get()

        # Disable button, show terminate button
        self.run_button.configure(state="disabled")
        self.terminate_button.pack(side=tk.RIGHT, padx=(0, 10))
        self.terminate_flag = False
        self.progress_text.set("0%")
        self.status_text.set("⏳ Working... this can take a bit.")

        # Get remix length
        remix_length = self.remix_length.get()
        
        # Get instrument toggles
        instrument_toggles = {
            'kick': self.enable_kick.get(),
            'snare': self.enable_snare.get(),
            'hihat': self.enable_hihat.get(),
            'bass': self.enable_bass.get(),
            'pitch': self.enable_pitch.get(),
            'melody': self.enable_melody.get(),
            'chords': self.enable_chords.get(),
            'vocals': self.enable_vocals.get(),
            'awesomeness': self.enable_awesomeness.get(),
            'epicness_pitch': self.enable_epicness_pitch.get(),
            # Zorammi chords only allowed when Zorammi Style is on
            'zorammi_chords': self.enable_zorammi_style.get(),
            'zorammi_style': self.enable_zorammi_style.get(),
        }

        self.render_thread = threading.Thread(
            target=self._run_generation,
            args=(video, output, bpm_val, quote_start, quote_end, remix_length, instrument_toggles),
            daemon=True
        )
        self.render_thread.start()

    def _run_generation(self, video: str, output: str, bpm: int, quote_start: float, quote_end: float, remix_length: int, instrument_toggles: dict):
        """Run the actual generation."""
        try:
            # Progress callback to update status
            def progress_callback(message: str):
                self.update_status(message)
            
            # Pass quote range, remix length, instrument toggles, and progress callback
            result = process_video(
                video_path=video, 
                output_path=output, 
                bpm=bpm,
                quote_start=quote_start,
                quote_end=quote_end,
                remix_length=remix_length,
                instrument_toggles=instrument_toggles,
                progress_callback=progress_callback
            )
            self.after(0, lambda r=result: self._handle_generation_result(r))
        except TypeError as te:
            # Fallback if process_video doesn't accept progress_callback yet
            if "progress_callback" in str(te):
                result = process_video(
                    video_path=video, 
                    output_path=output, 
                    bpm=bpm,
                    quote_start=quote_start,
                    quote_end=quote_end,
                    remix_length=remix_length,
                    instrument_toggles=instrument_toggles
                )
                self.after(0, lambda r=result: self._handle_generation_result(r))
            else:
                raise te
        except Exception as exc:
            self.after(0, lambda e=exc: self._handle_generation_error(e))
        finally:
            self.after(0, self._reset_after_generation)

    def _reset_after_generation(self):
        self.run_button.configure(state="normal")
        self.terminate_button.pack_forget()

    def _handle_generation_error(self, exc: Exception):
        if self.terminate_flag:
            self.status_text.set("⏹ Render terminated by user.")
            return
        self.progress_text.set("0%")
        self.status_text.set(f"❌ Error: {str(exc)[:50]}")
        messagebox.showerror("Error", f"Generation failed:\n{exc}")

    def _handle_generation_result(self, result):
        if result:
            p = Path(result)
            self.progress_text.set("100%")
            self.status_text.set(f"✅ Done! Saved to {p.name}")
            if p.suffix.lower() == ".mp4":
                self.after(100, lambda: self._update_output_preview_async(result))
            if p.suffix.lower() != ".mp4":
                log_hint = ""
                if _LOG_PATH is not None:
                    log_hint = f"\n\nLog saved to:\n{_LOG_PATH}"
                messagebox.showinfo("Success", f"Sparta Remix audio saved to:\n{result}\n\nVideo render may have been skipped or failed; check the console log.{log_hint}")
            else:
                messagebox.showinfo("Success", f"Sparta Remix saved to:\n{result}")
        else:
            self.status_text.set("⚠️ Something went wrong. See console.")

    def _terminate_render(self):
        """Terminate the current render."""
        self.terminate_flag = True
        self.status_text.set("⏹ Terminating render... (will stop after current operation)")

    def _update_output_preview(self, video_path: str):
        """Update the output preview canvas with the first frame of the output video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.output_canvas.update_idletasks()
                configured_w = int(float(self.output_canvas.cget("width") or 0))
                configured_h = int(float(self.output_canvas.cget("height") or 0))
                canvas_w = self.output_canvas.winfo_width() or configured_w
                canvas_h = self.output_canvas.winfo_height() or configured_h
                if canvas_w <= 1:
                    canvas_w = configured_w
                if canvas_h <= 1:
                    canvas_h = configured_h
                h, w = frame.shape[:2]
                scale = min(canvas_w/w, canvas_h/h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                
                img = Image.fromarray(frame)
                self.output_preview_image = ImageTk.PhotoImage(img)
                
                self.output_canvas.delete("all")
                x_offset = (canvas_w - new_w) // 2
                y_offset = (canvas_h - new_h) // 2
                self.output_canvas.create_image(x_offset, y_offset, 
                                                anchor=tk.NW, 
                                                image=self.output_preview_image)
                
                self.output_canvas.create_text(8, 8, 
                                              text="✓ Output Ready",
                                              fill=self.TURF_GREEN,
                                              font=("Segoe UI", 9, "bold"),
                                              anchor=tk.NW)
        except Exception as e:
            print(f"Error updating output preview: {e}")

    def _update_output_preview_async(self, video_path: str):
        def worker():
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.after(0, lambda f=frame: self._set_output_preview_frame(f))
            except Exception as e:
                print(f"Error updating output preview: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _set_output_preview_frame(self, frame):
        try:
            self.output_canvas.update_idletasks()
            configured_w = int(float(self.output_canvas.cget("width") or 0))
            configured_h = int(float(self.output_canvas.cget("height") or 0))
            canvas_w = self.output_canvas.winfo_width() or configured_w
            canvas_h = self.output_canvas.winfo_height() or configured_h
            if canvas_w <= 1:
                canvas_w = configured_w
            if canvas_h <= 1:
                canvas_h = configured_h
            h, w = frame.shape[:2]
            scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

            img = Image.fromarray(frame)
            self.output_preview_image = ImageTk.PhotoImage(img)

            self.output_canvas.delete("all")
            x_offset = (canvas_w - new_w) // 2
            y_offset = (canvas_h - new_h) // 2
            self.output_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.output_preview_image)
            self.output_canvas.create_text(
                8,
                8,
                text="✓ Output Ready",
                fill=self.TURF_GREEN,
                font=("Segoe UI", 9, "bold"),
                anchor=tk.NW,
            )
        except Exception as e:
            print(f"Error updating output preview: {e}")
    
    def update_status(self, message: str):
        """Thread-safe method to update status text from background thread."""
        def apply_update(msg: str):
            m = re.match(r"^\s*(\d{1,3})%\s*(.*)$", msg)
            if m:
                pct = int(m.group(1))
                pct = max(0, min(100, pct))
                self.progress_text.set(f"{pct}%")
                rest = (m.group(2) or "").strip()
                if rest:
                    self.status_text.set(rest)
                return
            self.status_text.set(msg)

        self.after(0, lambda m=message: apply_update(m))

    def destroy(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        super().destroy()


if __name__ == "__main__":
    app = ModernSpartaGUI()
    app.mainloop()
