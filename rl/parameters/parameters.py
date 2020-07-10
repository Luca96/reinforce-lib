
import math


class DynamicParameter:
    """Base class for dynamic (step-dependent) parameters"""
    def __init__(self, initial: float, final: float, steps: int, restart=False, decay_on_restart=None):
        assert isinstance(steps, int) and steps > 0
        assert isinstance(restart, bool)

        self.initial_value = initial
        self.final_value = final
        self.value = self.initial_value
        self.should_restart = restart
        self.should_decay_on_restart = isinstance(decay_on_restart, float)
        self.restart_decay_rate = decay_on_restart
        self.step_counter = 0
        self.max_steps = steps

    def __call__(self, *args, **kwargs) -> float:
        """Returns the (decayed) value of this parameter"""
        if self.step_counter == 0:
            self.step_counter += 1
            return self.initial_value

        if self.step_counter > self.max_steps:
            if self.should_restart:
                self.restart()
            else:
                return self.value

        self.value = self.compute_value()
        self.step_counter += 1
        return self.value

    def compute_value(self):
        raise NotImplementedError

    def restart(self):
        self.step_counter = 0

        if self.should_decay_on_restart:
            self.initial_value *= self.restart_decay_rate

        self.value = self.initial_value


class ExponentialParameter(DynamicParameter):
    """Exponential Parameter"""
    def __init__(self, rate: float, base=math.e, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base = base
        self.rate = 1 / (rate + 1e-7)
        self.delta = self.initial_value - self.final_value

    def compute_value(self, *args, **kwargs):
        t = self.step_counter / self.max_steps
        return self.delta * self.base**(-self.rate * t) + self.final_value

    def restart(self):
        super().restart()

        if self.should_decay_on_restart:
            self.delta = self.initial_value - self.final_value


class LinearParameter(DynamicParameter):
    """Linear Parameter"""
    def __init__(self, *args, rate=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.rate = rate
        self.decay_rate = (self.initial_value - self.final_value) / self.max_steps

    def compute_value(self, *args, **kwargs):
        k = self.rate**(self.step_counter / self.max_steps)
        return self.decay_rate * k * (self.max_steps - self.step_counter) + self.final_value

    def restart(self):
        super().restart()

        if self.should_decay_on_restart:
            self.decay_rate = (self.initial_value - self.final_value) / self.max_steps


class StepParameter(DynamicParameter):
    """Step-Parameter (can be seen as a constant-parameter as special case)"""
    def __init__(self, value: float, *args, **kwargs):
        super().__init__(*args, initial=value, final=value, **kwargs)

    def compute_value(self):
        return self.value
