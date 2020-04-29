import torch as pt

pt.__version__

LEARNING_RATE = .03


# Scalar class
class Scalar:
    def __init__(self, val):
        self.val = val
        self.grad = 0.
        self.backward = lambda: None

    def __add__(self, other):
        out = Scalar(self.val + other.val)

        def backward():
            self.grad += out.grad
            other.grad += out.grad
            self.backward(), other.backward()

        out.backward = backward
        return out

    def __mul__(self, other):
        out = Scalar(self.val * other.val)

        def backward():
            self.grad += out.grad * other.val
            other.grad += out.grad * self.val
            self.backward(), other.backward()

        out.backward = backward
        return out

    def __repr__(self):
        return f"Value: {self.val}, Grad: {self.grad}"


def loss(y_pred, y):
    error = [y_pred[i] + Scalar(-1.) * y[i] for i in range(len(y))]
    squared_error = [error[i] * error[i] for i in range(len(error))]
    sum_squared_errors = sum(squared_error, Scalar(0.))
    mean_squared_errors = sum_squared_errors * Scalar(1. / len(squared_error))
    return mean_squared_errors


def forward(w, X):
    return [w * X[i] for i in range(len(X))]


if __name__ == '__main__':
    # Apply `Scalar` to linear regression
    ptX = pt.linspace(-5, 5, 10)
    pty = 5 * ptX + pt.randn(len(ptX))

    # Make linear regression data
    X = [Scalar(x.item()) for x in ptX]
    y = [Scalar(y.item()) for y in pty]

    # Implement the mean squared error calculation
    w = pt.rand(1).item()
    w = Scalar(w)

    y_pred = forward(w, X)

    for _ in range(5):
        y_pred = forward(w, X)

        mean_squared_errors = loss(y_pred, y)

        w.grad = 0.
        mean_squared_errors.grad = 1.

        mean_squared_errors.backward()
        w.val -= LEARNING_RATE * w.grad
        print(w)
