# Challenge the AI (i.e., try to beat Level 1-1 under 36.18 seconds)
# Press SELECT to start a game; SELECT and START to exit the script
import time
from pathlib import Path

from pyboy import PyBoy

from challenge import NESController

controller = NESController()
# Start polling at 60Hz (default)
controller.start_polling()

# initialize emulator
rom = Path("game/Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc")
video = Path("ai-run.mp4")
pyboy = PyBoy(str(rom), sound=False, window="SDL2", scale=2.5)
quit_game = False

while not quit_game:
    # press SELECT -> to start a game
    if controller.select:
        # TODO: startfile(video)

        # load initial state
        with Path("state/level1-1.state").open("rb") as f:
            pyboy.load_state(f)

        game_over = False

        start_time = time.time()
        while not game_over and not quit_game:
            # Press SELECT and START -> to exit the loop and quit the script
            quit_game = controller.select and controller.start

            pyboy.tick()

            is_dead = pyboy.memory[0xC1C1] == 3
            on_map = pyboy.memory[0xC1C1] == 4

            game_over = is_dead or on_map

            flag_reached = pyboy.memory[0xC1C2] == 12

            # TODO: quit the video

            if flag_reached:
                end_time = time.time()
                seconds = end_time - start_time

                if seconds < 43.4:
                    print("You beat the AI!")

                print(f"Flag reached in {seconds} seconds")
                game_over = True

pyboy.stop()
