import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data ={
    "experience": [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5],
    "education": ['BS', 'BS', 'MS', 'BS', 'MS', 'PhD', 'MS', 'PhD', 'BS', 'PhD', 'BS', 'MS'],
    "salary_k": [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]
}
df = pd.DataFrame(data)

print("Missing values before:")
print(df.isnull().sum())
print("\n")
df["experience"] = df["experience"].fillna(df["experience"].median())

df = pd.get_dummies(df, columns=["education"], drop_first=True)
print("Columns after encoding:", list(df.columns))

X = df.drop("salary_k", axis=1)
y = df["salary_k"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Training rows: ", len(X_train))
print("Testing rows: ", len(X_test))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("R2:", round(r2_score(y_test, y_pred), 3))

ms_graduate = pd.DataFrame([[4, True, False]], columns=X.columns)
salary = model.predict(scaler.transform(ms_graduate))[0]
print("Predicted salary of MS graduate with 4 years experience:", round(salary, 2), "k")