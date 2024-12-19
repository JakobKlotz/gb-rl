import threading
import time

import hid


class NESController:
    def __init__(self):
        """miadore NES Controller (USB)"""
        self.gamepad = hid.device()
        self.gamepad.open(0x0079, 0x0011)
        self.gamepad.set_nonblocking(True)

        # Button states
        self.a = False
        self.b = False
        self.start = False
        self.select = False
        self.left = False
        self.right = False
        self.up = False
        self.down = False

        # Control flags
        self._running = False
        self._polling_thread = None
        self._polling_rate = 1 / 60  # 60Hz polling rate by default

    def get_state(self) -> list:
        state = self.gamepad.read(64)

        if state:
            self.a = (state[5] & 0b00100000) > 0
            self.b = (state[5] & 0b01000000) > 0
            self.start = (state[6] & 0b00100000) > 0
            self.select = (state[6] & 0b00010000) > 0
            self.left = state[3] == 0
            self.right = state[3] == 255
            self.up = state[4] == 0
            self.down = state[4] == 255

        return state

    def _polling_loop(self) -> None:
        """Background polling loop"""
        while self._running:
            self.get_state()
            time.sleep(self._polling_rate)

    def start_polling(self, polling_rate: float = 1 / 60) -> None:
        """Start continuous polling in background thread

        Args:
            polling_rate (float): Time between polls in seconds (default: 1/60)
        """
        if self._polling_thread is None or not self._polling_thread.is_alive():
            self._polling_rate = polling_rate
            self._running = True
            self._polling_thread = threading.Thread(target=self._polling_loop)
            self._polling_thread.daemon = (
                True  # Thread will exit when main program exits
            )
            self._polling_thread.start()

    def stop_polling(self) -> None:
        """Stop the background polling"""
        self._running = False
        if self._polling_thread:
            self._polling_thread.join()
            self._polling_thread = None
