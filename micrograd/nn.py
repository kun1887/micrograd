import random
from micrograd.engine import Value


class neuron:

    def __init__(self, input_dim, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_dim)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = Value(0)
        for wi, xi in zip(self.w, x):
            act += wi * xi
        act += self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        activation = "relu" if self.nonlin else "linear"
        return f"Neuron({len(self.w)}) with {activation} activation"


class Layer:

    def __init__(self, input_dim, output_dim, **kwargs):
        self.layer = [neuron(input_dim, **kwargs) for _ in range(output_dim)]

    def __call__(self, x):
        out = [n(x) for n in self.layer]
        return out

    def parameters(self):
        return [p for n in self.layer for p in n.parameters()]

    def __repr__(self):
        return f"layer of {len(self.layer)} neurons: {[n for n in self.layer]}"
