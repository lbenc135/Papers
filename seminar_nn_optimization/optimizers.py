from autograd import *
import numpy as np


class SGD:
    x0 = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.0001):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        # update
        self.x0 = self.x0 - self.learning_rate * dz_dx

        return self.x0


class Momentum:
    x0 = np.array([0., 0.])
    v = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.0001, gamma=0.9):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr
        self.gamma = gamma

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        # momentum
        self.v = self.gamma * self.v + self.learning_rate * dz_dx

        # update
        self.x0 = self.x0 - self.v

        return self.x0


class NAG:
    x0 = np.array([0., 0.])
    v = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.0001, gamma=0.9):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr
        self.gamma = gamma

    def step_minimize(self):
        # gradients
        x = self.x0[0] - self.gamma * self.v[0]
        y = self.x0[1] - self.gamma * self.v[1]
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(x, y),
                          elementwise_grad(self.z, argnum=1)(x, y)])

        # momentum
        self.v = self.gamma * self.v + self.learning_rate * dz_dx

        # update
        self.x0 = self.x0 - self.v

        return self.x0


class Adagrad:
    x0 = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.1, epsilon=1e-8):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr
        self.gti = np.zeros(x0.shape[0])
        self.epsilon = epsilon

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        # learning rate tuning
        self.gti += dz_dx ** 2
        lr = self.learning_rate / (self.epsilon + np.sqrt(self.gti))

        # update
        self.x0 = self.x0 - lr * dz_dx

        return self.x0


class Adadelta:
    x0 = np.array([0., 0.])

    def __init__(self, z, x0, learning_rate=0.01, gamma=0.9, epsilon=1e-8):
        self.z = z
        self.x0 = np.copy(x0)
        self.Edwt = np.ones(x0.shape[0]) * learning_rate
        self.Egt = np.zeros(x0.shape[0])
        self.gamma = gamma
        self.epsilon = epsilon

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        # update calculation
        self.Egt = self.gamma * self.Egt + (1 - self.gamma) * (dz_dx ** 2)
        dw = (np.sqrt(self.Edwt) + self.epsilon) / (np.sqrt(self.Egt) + self.epsilon) * dz_dx
        self.Edwt = self.gamma * self.Edwt + (1 - self.gamma) * (dw ** 2)

        # update
        self.x0 = self.x0 - dw

        return self.x0


class Adam:
    x0 = np.array([0., 0.])
    m = np.array([0., 0.])
    v = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        # momentum
        self.m = self.beta1 * self.m + (1 - self.beta1) * dz_dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * dz_dx ** 2
        # bias correction
        m = self.m / (1 - self.beta1)
        v = self.v / (1 - self.beta2)

        # update
        self.x0 = self.x0 - self.learning_rate / (np.sqrt(v) + self.epsilon) * m

        return self.x0


class Nadam:
    x0 = np.array([0., 0.])
    m = np.array([0., 0.])
    v = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.001, beta1=0.9, beta2=0.999, gamma=0.9, epsilon=1e-8):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.epsilon = epsilon

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        # momentum
        self.m = self.beta1 * self.m + (1 - self.beta1) * dz_dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * dz_dx ** 2
        # bias correction
        v = self.v / (1 - self.beta2)

        # update
        self.x0 = self.x0 - self.learning_rate / (np.sqrt(v) + self.epsilon) * (
                    self.beta1 * self.m / (1 - self.beta1) + (1 - self.beta1) * dz_dx / (1 - self.beta1))

        return self.x0

class AMSGrad:
    x0 = np.array([0., 0.])
    m = np.array([0., 0.])
    v = np.array([0., 0.])

    def __init__(self, z, x0, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.z = z
        self.x0 = np.copy(x0)
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step_minimize(self):
        # gradients
        dz_dx = np.array([elementwise_grad(self.z, argnum=0)(self.x0[0], self.x0[1]),
                          elementwise_grad(self.z, argnum=1)(self.x0[0], self.x0[1])])

        v_last = self.v / (1 - self.beta2)

        # momentum
        self.m = self.beta1 * self.m + (1 - self.beta1) * dz_dx
        self.v = self.beta2 * self.v + (1 - self.beta2) * dz_dx ** 2
        # bias correction
        m = self.m / (1 - self.beta1)
        v_new = self.v / (1 - self.beta2)
        v = [max(v_last[0], v_new[0]), max(v_last[1], v_new[1])]

        # update
        self.x0 = self.x0 - self.learning_rate / (np.sqrt(v) + self.epsilon) * m

        return self.x0


# f = lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
# x0 = np.array([2., 2.])
#
# nadam = Nadam(f, x0, 0.001)
# res = [x0]
# training_steps = 1000
# for j in range(0, training_steps):
#     xy = np.copy(nadam.step_minimize())
#     res.append(xy)
