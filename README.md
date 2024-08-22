# Micrograd

Implementation from scratch of a tiny scalar-only Autograd engine and a simple neural netword library built on top of the engine.

The engine provides a backward() method to backpropagate gradient from a scalar L to all the other scalars used to compute L.

The neural net library provides a neuron and layer implementation with relu or linear activation.

### example of use case:

```python
from micrograd.engine import Value

a = Value(10)
b = Value(20)
c = (a*b).relu()
d = Value(30)
L = c + d
L.backward()
print(b) #prints Value(data=20, grad=10)
print(a) #prints Value(data=10, grad=20)
