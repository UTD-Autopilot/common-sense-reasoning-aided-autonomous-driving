
class Message():
    pass

class ExitMessage(Message):
    pass

class ScheduleCallMessage(Message):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
