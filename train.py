import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

learning_rate = 0.1
nb_iter = 1000

def visualize(X, Y, a, b):
    plt.title('Price / Mileage')
    plt.ylabel('Price')
    plt.xlabel('Mileage')
    plt.scatter(X, Y)
    plt.plot(X, a*X+b, color='red')
    plt.show()

def prompt_loss(RMSE):
    print('----------------------------------------')
    ploted = []
    for i in range(len(RMSE)):
        if i % 10 == 0:
            print('\rRMSE =', RMSE[i], end='')
            ploted.append(RMSE[i])
            plt.clf()
            plt.title(f'Root Mean Squared Error / iterations\ncurrent RMSE = {RMSE[i]:.3} (iteration ' + str(i) + ')')
            plt.ylabel('RMSE')
            plt.xlabel('Iterations')
            plt.plot([i*10 for i in range(len(ploted))], ploted)
            plt.pause(0.1)
    plt.show()
    print()

a = 0
b = 0


df = pd.read_csv('data.csv')
X = np.array(df['km'])
Y = np.array(df['price'])
coef_X = X.max()
coef_Y = Y.max()
# scaling
X = X / coef_X
Y = Y / coef_Y

RMSE = [] # Root Mean Squared Error Tab

# learning...
for i in range(nb_iter):
    grad_b = (a * X + b - Y).mean()
    grad_a = ((a * X + b - Y) * X).mean()
    loss = ((a * X + b - Y) **2 ).mean()
    RMSE.append(loss**0.5 * coef_Y)

    b -= learning_rate * grad_b
    a -= learning_rate * grad_a

loss = ((a * X + b - Y) **2 ).mean()
RMSE.append(loss**0.5 * coef_Y)

# sauvegarde des donnees dans result.csv
resfile = pd.read_csv('result.csv')
resfile['theta0'] = b * coef_Y
resfile['theta1'] = a * coef_Y / coef_X
resfile.to_csv('result.csv', index=False)

def bonus():
    print('MSE =', ((a * X + b - Y) **2 ).mean() * coef_Y ** 2)
    print('RMSE =', ((a * X + b - Y) **2 ).mean()**0.5 * coef_Y)
    # pour avoir une idee plus precise (qui ne depend plus de l'unité -> %), on peut calculer la RSE (Relative Squared Error)
    print('RSE =', ((a * X + b - Y) **2 ).mean() / ((Y - Y.mean())**2).mean())

    visualize(X * coef_X, Y * coef_Y, a * coef_Y / coef_X, b * coef_Y)
    prompt_loss(RMSE)


bonus()
