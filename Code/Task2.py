import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
y = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])


def gradient_descent(x, y, lr, epochs):
    m = 0.0
    c = 0.0
    n = len(x)
    losses = []
    for epoch in range(1, epochs + 1):
        y_hat = m * x + c
        error = y_hat - y
        dm = (2 / n) * np.sum(error * x)
        dc = (2 / n) * np.sum(error)
        m = m - lr * dm
        c = c - lr * dc
        mse = np.mean((y - (m * x + c)) ** 2)
        losses.append(mse)
        if epoch % 100 == 0:
            print("  epoch", epoch, "MSE =", mse)
    return m, c, losses


epochs = 1000
rates = [0.001, 0.01, 0.05]
results = {}

with np.errstate(all="ignore"):
    for lr in rates:
        print("Learning rate:", lr)
        m, c, losses = gradient_descent(x, y, lr, epochs)
        results[lr] = (m, c, losses)
        print()

plt.figure(figsize=(9, 5))
for lr in rates:
    plt.plot(range(1, epochs + 1), results[lr][2], label="lr = " + str(lr))
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("Loss Curve for Different Learning Rates")
plt.legend()
plt.show()

sk = LinearRegression().fit(x.reshape(-1, 1), y)
print("scikit-learn : m =", round(sk.coef_[0], 4), " c =", round(sk.intercept_, 4))
for lr in rates:
    m, c, losses = results[lr]
    print("lr =", lr, ": m =", round(m, 4), " c =", round(c, 4), " final MSE =", losses[-1])

print()
for lr in rates:
    final = results[lr][2][-1]
    if not np.isfinite(final) or final > 1e6:
        print("lr =", lr, "-> diverged, step size too large")
    elif abs(results[lr][0] - sk.coef_[0]) < 0.01:
        print("lr =", lr, "-> converged to the scikit-learn solution")
    else:
        print("lr =", lr, "-> still converging, needs more epochs")