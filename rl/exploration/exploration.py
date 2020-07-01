
class ExplorationMethod:
    def bonus(self, state):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class NoExploration(ExplorationMethod):

    def bonus(self, state) -> float:
        return 0.0

    def train(self):
        pass
