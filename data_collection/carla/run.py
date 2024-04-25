import threading

from autopilot.ui import AutopilotGUI
import jurigged

jurigged.watch("autopilot")

if __name__ == '__main__':
    gui = AutopilotGUI()  # old ui
