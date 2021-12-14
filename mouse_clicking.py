from pynput.mouse import Controller, Button

import time

mouse = Controller()

while True:
    mouse.click(Button.right, 1)
    time.sleep(180)