from pathlib import Path

from pyboy import PyBoy
from pyxboxcontroller import XboxController

controller = XboxController(0)

rom = Path("game/Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc")

pyboy = PyBoy(str(rom), sound=False, window="SDL2", scale=5)
quit_game = False

while not quit_game:
    # press a to start a game
    if controller.state.a:
        # load initial state
        with Path("state/level1-1.state").open("rb") as f:
            pyboy.load_state(f)

        game_over = False
        while not game_over and not quit_game:
            quit_game = controller.state.start

            pyboy.tick()
            is_dead = pyboy.memory[0xC1C1] == 3
            on_map = pyboy.memory[0xC1C1] == 4

            game_over = is_dead or on_map

            if game_over:
                print("Mario is dead")

            if pyboy.memory[0xC1C2] == 12:
                print("Flag reached")

pyboy.stop()
