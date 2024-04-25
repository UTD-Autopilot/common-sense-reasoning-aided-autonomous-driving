import imgui


class Window(object):
    def __init__(self):
        self.closed = False

    def render(self):
        pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.on_close()

    def on_close(self):
        pass
