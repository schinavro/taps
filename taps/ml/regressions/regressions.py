

class Regression:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        self.setup(*args, **kwargs)
        self.train(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass
