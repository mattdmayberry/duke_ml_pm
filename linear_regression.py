import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mae_percent
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# delete results file contents
open('results.txt', 'w').close()

df = pd.read_csv('CCPP_data.csv')
cols = ['Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity', 'Net Hourly Electrical Energy Output']
df.columns = cols


def split_data(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state, test_size=0.2)
    return X_train, X_test, y_train, y_test

def save_results(name, model_mse, model_mae, y, columns, coefs, random_state):
    results = open("results.txt", "a")
    results.write(f'Model: {name}\n')
    results.write(f'Dependent Variable: {y}\n')
    results.write(f'Random State: {random_state}\n')
    results.write("Features: ")
    for i in columns:
        results.write(f'{i}, ')
    results.write("\n")
    results.write("Coefficients: ")
    for j in coefs:
        results.write(f'{j}, ')
    results.write("\n")
    results.write(f'MSE: {model_mse}\n')
    results.write(f'MAE %: {model_mae}\n')
    results.write("\n")


def plot_coefficients(columns, coefs):
    plt.figure(figsize=(12, 6))
    plt.barh(columns, coefs)
    plt.xlabel("Coefficients")
    plt.ylabel("Features")
    plt.title("Feature Coefficients")
    plt.show()

def simple_linear_regression(data, random_state):
    name = "Simple Linear Regression"
    model = LinearRegression()
    X = data.iloc[:, :-4]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_data(X, y, random_state)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    model_mse = mse(y_test, y_hat)
    model_mae = mae_percent(y_test, y_hat)
    save_results(name, model_mse, model_mae, y.head().name, X_test.columns, model.coef_, random_state)
    # plot_coefficients(X_test.columns, model.coef_)

def multiple_linear_regression(data, random_state):
    name = "Multiple Linear Regression"
    model = LinearRegression()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_data(X, y, random_state)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    model_mse = mse(y_test, y_hat)
    model_mae = mae_percent(y_test, y_hat)
    save_results(name, model_mse, model_mae, y.head().name, X_test.columns, model.coef_, random_state)
    # plot_coefficients(X_test.columns, model.coef_)

# create single linear regressions with tuning of random state hyperparameter
simple_linear_regression(df, 0)
simple_linear_regression(df, 42)

# create multiple linear regressions with tuning of random state hyperparameter
multiple_linear_regression(df, 0)
multiple_linear_regression(df, 42)


def visualize_data(df):
    # visualize variable correlations using Seaborn heatmap
    fig, ax = plt.subplots(figsize=(15, 15))
    dataplot = sns.heatmap(df.corr(), annot=True)
    plt.show()

    # visualize data using pyplot scatter matrix
    scatter_matrix(df, color='black', figsize=(12, 12), hist_kwds={'color': 'black'})
    plt.show()

# visualize_data(df)
