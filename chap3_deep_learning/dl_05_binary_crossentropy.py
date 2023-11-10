import numpy as np
import matplotlib.pyplot as plt


class BinaryCrossEntropy:
    def __init__(self):
        self.one = lambda y_hat: -np.log(y_hat)
        self.zero = lambda y_hat: -np.log(1-y_hat)

    def forward(self, y, y_hat):
        if y == 1:
            return self.one(y_hat)

        return self.zero(y_hat)


bce = BinaryCrossEntropy()
print(f"{bce.forward(0, 0.3)}")


class BCELoss:
    def forward(self, y, pred):
        J = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        return J


loss_function = BCELoss()
J = loss_function.forward(y=0, pred=0.3)
print(f"y=0, pred=0.3 -> J = {J:.4f}")

preds = np.arange(start=0.1, stop=1, step=0.01)
case1 = [loss_function.forward(y=0, pred=i) for i in preds]
case2 = [loss_function.forward(y=1, pred=i) for i in preds]

fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
axes[0].plot(preds, case1)
axes[0].set_title('case.1 y=0'); axes[0].set_xlabel('preds'); axes[0].set_ylabel('loss')

axes[1].plot(preds, case2)
axes[1].set_title('case.2 y=1'); axes[1].set_xlabel('preds'); axes[1].set_ylabel('loss')

plt.show()

