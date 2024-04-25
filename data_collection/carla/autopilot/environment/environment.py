
class Environment():
    def __init__(self):
        pass
    
    def connect(self):
        """Connect to the environment."""
        raise NotImplementedError()
    
    def disconnect(self):
        """Disconnect from the environment and clean up resources."""
        return
    
class Recorder():
    def __init__(self):
        pass

    def record_start(self):
        raise NotImplementedError()
    
    def record_stop(self):
        raise NotImplementedError()

class DummyControl():
    def __init__(self):
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.reverse = False
        self.hand_brake = False
