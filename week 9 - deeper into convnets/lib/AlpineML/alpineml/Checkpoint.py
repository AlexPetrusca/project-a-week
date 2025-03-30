import pickle as pkl

class Checkpoint:
    def __init__(self, params=None):
        self.params = params

    def write(self, path):
        with open(path, "wb") as file:
            pkl.dump(self.params, file)

    @staticmethod
    def read(path):
        ckp = Checkpoint()
        with open(path, "rb") as file:
            ckp.params = pkl.load(file)
        return ckp