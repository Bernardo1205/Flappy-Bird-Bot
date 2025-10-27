# counter.py
import sys
import time
from multiprocessing import Process, Queue
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QFont, QColor, QPalette


def _run_counter(queue: Queue):
    """Run the transparent overlay in its own process."""
    app = QApplication.instance() or QApplication(sys.argv)

    label = QLabel("Steps: 0")
    label.setWindowFlags(
        Qt.FramelessWindowHint
        | Qt.WindowStaysOnTopHint
        | Qt.Tool
    )
    label.setAttribute(Qt.WA_TranslucentBackground, True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    label.setMargin(10)
    label.move(30, 30)

    font = QFont("Arial", 20, QFont.Bold)
    label.setFont(font)
    label.setStyleSheet("color: lime;")

    # fully transparent background
    palette = label.palette()
    palette.setColor(label.backgroundRole(), QColor(0, 0, 0, 0))
    label.setPalette(palette)

    label.show()

    last_value = 0

    def update_label():
        nonlocal last_value
        while not queue.empty():
            last_value = queue.get_nowait()
        label.setText(f"Steps: {last_value}")

    timer = QTimer()
    timer.timeout.connect(update_label)
    timer.start(100)  # update every 100 ms

    app.exec()


class StepCounter:
    """Starts the overlay in another process and sends it step updates."""
    def __init__(self):
        self.queue = Queue()
        self.process = None

    def start(self):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=_run_counter, args=(self.queue,), daemon=True)
            self.process.start()

    def set_steps(self, n: int):
        if self.process and self.process.is_alive():
            self.queue.put(n)

    def increment(self, delta: int = 1):
        if self.process and self.process.is_alive():
            # We can keep only the latest value by sending absolute count
            self.queue.put(delta)

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)


