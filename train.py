import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

learning_rate = 0.1
nb_iter = 1000

def visualize(X, Y, a, b):
    plt.title('Root Mean Squared Error according to its mileage')
    plt.scatter(X, Y)
    plt.plot(X, a*X+b)
    plt.show()

def prompt_loss(RMSE):
    plt.title('Root Mean Squared Error / iterations')
    plt.plot(RMSE)
    plt.show()

a = 0
b = 0


df = pd.read_csv('data.csv')
X = np.array(df['km'])
Y = np.array(df['price'])
coef_X = X.max()
coef_Y = Y.max()
X = X / coef_X
Y = Y / coef_Y

RMSE = [] # Root Mean Squared Error Tab

# learning...
for i in range(nb_iter):
    grad_b = (a * X + b - Y).mean()
    grad_a = ((a * X + b - Y) * X).mean()
    loss = ((a * X + b - Y) **2 ).mean()

    b -= learning_rate * grad_b
    a -= learning_rate * grad_a
    RMSE.append(loss**0.5 * coef_Y)

def bonus():
    print('MSE =', ((a * X + b - Y) **2 ).mean() * coef_Y ** 2)
    print('RMSE =', ((a * X + b - Y) **2 ).mean()**0.5 * coef_Y)
    # pour avoir une idee plus precise (qui ne depend plus de l'unité -> %), on peut calculer la RSE (Relative Squared Error)
    print('RSE =', ((a * X + b - Y) **2 ).mean() / ((Y - Y.mean())**2).mean())

    visualize(X * coef_X, Y * coef_Y, a * coef_Y / coef_X, b * coef_Y)
    prompt_loss(RMSE)

# sauvegarde des donnees dans result.csv
resfile = pd.read_csv('result.csv')
resfile['theta0'] = b * coef_Y
resfile['theta1'] = a * coef_Y / coef_X
resfile.to_csv('result.csv', index=False)

bonus()
