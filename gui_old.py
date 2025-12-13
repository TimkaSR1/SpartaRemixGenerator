import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from main import process_video
from config import OUTPUT_DIR


class SpartaGUI(tk.Tk):
    """Simple GUI wrapper around the CLI Sparta Remix generator."""

    def __init__(self):
        super().__init__()
        self.title("Sparta Remix Generator")
        self.geometry("520x300")
        self.resizable(False, False)

        self.video_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(OUTPUT_DIR / "sparta_remix_gui.wav"))
        self.bpm = tk.StringVar(value="140")
        self.status_text = tk.StringVar(value="Ready.")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Video selector
        tk.Label(self, text="Input video:").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.video_path, width=45).grid(row=0, column=1, **pad)
        tk.Button(self, text="Browse", command=self._browse_video).grid(row=0, column=2, **pad)

        # Output selector
        tk.Label(self, text="Output wav:").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.output_path, width=45).grid(row=1, column=1, **pad)
        tk.Button(self, text="Browse", command=self._browse_output).grid(row=1, column=2, **pad)

        # BPM
        tk.Label(self, text="BPM:").grid(row=2, column=0, sticky="w", **pad)
        tk.Entry(self, textvariable=self.bpm, width=10).grid(row=2, column=1, sticky="w", **pad)

        # Status
        tk.Label(self, textvariable=self.status_text, fg="blue").grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        # Action button
        self.run_button = tk.Button(self, text="Generate Sparta Remix", command=self._run_async, width=25)
        self.run_button.grid(row=4, column=0, columnspan=3, **pad)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if path:
            self.output_path.set(path)

    def _run_async(self):
        """Run generation on a background thread to keep UI responsive."""
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

        output = self.output_path.get().strip() or str(OUTPUT_DIR / "sparta_remix_gui.wav")

        # Disable button during processing
        self.run_button.config(state=tk.DISABLED)
        self.status_text.set("Working... this can take a bit.")

        thread = threading.Thread(
            target=self._run_generation,
            args=(video, output, bpm_val),
            daemon=True
        )
        thread.start()

    def _run_generation(self, video: str, output: str, bpm: int):
        try:
            result = process_video(video_path=video, output_path=output, bpm=bpm)
            if result:
                self.status_text.set(f"Done! Saved to {result}")
                messagebox.showinfo("Success", f"Sparta Remix saved to:\n{result}")
            else:
                self.status_text.set("Something went wrong. See console for details.")
        except Exception as exc:
            self.status_text.set("Error during generation.")
            messagebox.showerror("Error", f"Generation failed:\n{exc}")
        finally:
            self.run_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    app = SpartaGUI()
    app.mainloop()
