# Tkinter GUI to display the game and the AI run side by side
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
from pyboy import PyBoy

from challenge import NESController

FONT = ("Cascadia Mono", 18)
RESOLUTION = (160, 144)
SCALE = 3


class MarioChallenge:
    def __init__(self, root):
        # resolution of video and emulator
        width, height = RESOLUTION
        self.width = width * SCALE
        self.height = height * SCALE

        self.root = root
        self.root.title("MC(A)I Mario Challenge")

        # Create a static text label at the top
        self.text_label = tk.Label(
            root, text="Press SELECT to start a game", font=FONT
        )
        self.text_label.pack(fill=tk.X, padx=10, pady=5)

        # Create main container
        self.container = ttk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create frames for emulator and video
        self.emu_frame = ttk.LabelFrame(self.container, text="Your game")
        self.emu_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.video_frame = ttk.LabelFrame(self.container, text="The AI run")
        self.video_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Configure grid weights
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=1)

        # Initialize emulator components
        self.emu_canvas = tk.Canvas(
            self.emu_frame, width=self.width, height=self.height
        )
        self.emu_canvas.pack(padx=0, pady=0)

        self.control_frame = ttk.Frame(self.emu_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Initialize video components
        self.video_canvas = tk.Canvas(
            self.video_frame, width=self.width, height=self.height
        )
        self.video_canvas.pack(padx=0, pady=0)

        self.video_control_frame = ttk.Frame(self.video_frame)
        self.video_control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create label for game status
        self.status_label = tk.Label(root, text="", font=FONT)
        self.status_label.pack()

        # Initialize emulator and video state
        self.pyboy = None
        self.cap = None
        self.video_playing = False
        self.controller = None
        self.game_over = False
        self.flag_reached = False
        # keep track of time for each run
        self.start_time = None
        self.elapsed_time = None

        self.emulator_update_task = None
        self.video_update_task = None

    def init_emulator(self, rom_path):
        """Initialize the PyBoy emulator with the specified ROM"""
        self.pyboy = PyBoy(rom_path, window="SDL2")
        self.pyboy.set_emulation_speed(1)

    def update_emulator(self):
        """Update the emulator display"""
        if self.pyboy:
            # Get the screen buffer as RGB array
            screen = self.pyboy.screen.ndarray[:, :, :3]

            # Convert numpy array to PIL Image
            screen = Image.fromarray(screen)
            screen = screen.resize((self.width, self.height))
            screen = ImageTk.PhotoImage(screen)

            # Update canvas
            self.emu_canvas.create_image(0, 0, anchor=tk.NW, image=screen)
            self.emu_canvas.image = screen  # Keep reference

            # Tick the emulator
            if not self.game_over:
                self.pyboy.tick()

            # Check game state
            self.check_game_state()

            if self.game_over:
                if self.elapsed_time < 43.4 and self.flag_reached:
                    self.status_label.config(text="You beat the AI!")
                elif self.elapsed_time >= 43.4 and self.flag_reached:
                    self.status_label.config(text="You were too slow!")
                else:
                    self.status_label.config(text="Game over!")

            if not self.game_over:
                # Schedule next update
                self.emulator_update_task = self.root.after(
                    16, self.update_emulator
                )  # ~60 FPS

    def load_state(self):
        """Load a save state"""
        if self.pyboy:
            with Path("state/luigi-level1-1.state").open("rb") as f:
                self.pyboy.load_state(f)

    def init_controller(self):
        self.controller = NESController()
        self.controller.start_polling()

    def run(self):
        if self.controller.select and not self.video_playing:
            self.load_state()
            self.reset_video()
            self.video_playing = True

            if self.game_over:
                self.game_over = False
                self.status_label.config(text="")

            # Record start time
            self.start_time = time.time()

            # Cancel any existing scheduled tasks
            if self.emulator_update_task:
                self.root.after_cancel(self.emulator_update_task)
            if self.video_update_task:
                self.root.after_cancel(self.video_update_task)

            self.update_video()
            self.update_emulator()

        self.root.after(8, self.run)

    def restart_video(self):
        """Restart the video playback"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def init_video(self, video_path):
        """Initialize video playback"""
        self.cap = cv2.VideoCapture(video_path)

    def reset_video(self):
        """Reset video playback"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_playing = False

    def update_video(self):
        """Update video frame"""
        if self.cap and self.video_playing:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize((self.width, self.height))
                frame = ImageTk.PhotoImage(frame)

                # Update canvas
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.video_canvas.image = frame

                # Schedule next frame
                self.video_update_task = self.root.after(
                    16, self.update_video
                )  # ~60 FPS
            else:
                # Video finished, reset
                self.reset_video()

    def toggle_video(self):
        """Toggle video playback"""
        if self.cap:
            self.video_playing = not self.video_playing

            if self.video_playing:
                self.update_video()

    def check_game_state(self):
        """Check the game state and handle video playback"""
        if self.pyboy:
            is_dead = self.pyboy.memory[0xC1C1] == 3
            on_map = self.pyboy.memory[0xC1C1] == 4
            flag_reached = self.pyboy.memory[0xC1C2] == 12

            if is_dead or on_map or flag_reached:
                self.video_playing = False
                self.game_over = True
                self.flag_reached = flag_reached

                # Calculate elapsed time
                self.elapsed_time = time.time() - self.start_time


def main():
    root = tk.Tk()
    app = MarioChallenge(root)

    app.init_controller()
    # Initialize emulator with ROM
    rom_path = "game/Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc"
    app.init_emulator(rom_path)

    # Initialize video
    video_path = "challenge/assets/ai-run.mp4"
    app.init_video(video_path)

    app.run()
    root.mainloop()


if __name__ == "__main__":
    main()
