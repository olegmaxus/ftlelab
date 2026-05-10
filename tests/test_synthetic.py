import ftlelab as ftle
from ftlelab.data import make_moons_dataset, make_circle_dataset, make_spiral_dataset, make_xor_dataset
import matplotlib.pyplot as plt

X, y = make_moons_dataset()
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='bwr', alpha=0.5)
plt.title("Moons Dataset")
plt.show()

X, y = make_circle_dataset()
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='bwr', alpha=0.5)
plt.title("Circle Dataset")
plt.show()

X, y = make_spiral_dataset()
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='bwr', alpha=0.5)
plt.title("Spiral Dataset")
plt.show()

X, y = make_xor_dataset()
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='bwr', alpha=0.5)
plt.title("XOR Dataset")
plt.show()