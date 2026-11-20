import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
marks = [32, 40, 41, 50, 54, 60, 62, 70, 78, 88]

X = np.array(hours).reshape(-1, 1)
y = np.array(marks)

model = LinearRegression()
model.fit(X, y)

print("Slope:", round(model.coef_[0], 4))
print("Intercept: ", round(model.intercept_, 4))

for h in [5.5, 8]:
    print("Predicted marks for", h, "hours:", round(model.predict([[h]])[0], 2))
y_pred = model.predict(X)

plt.scatter(hours, marks, color="blue", label="Data points")
plt.plot(hours, y_pred, color="red", label="Regression line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Study Hours vs Marks")
plt.legend()
plt.show()

print("MAE:", round(mean_absolute_error(y, y_pred), 4))
print("MSE:", round(mean_squared_error(y, y_pred), 4))
print("R2:", round(r2_score(y, y_pred), 4))