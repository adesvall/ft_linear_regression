import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

learning_rate = 0.2
nb_iter = 1000

a = 0
b = 0

try:
    df = pd.read_csv('data.csv')
    X = np.array(df['km'])
    Y = np.array(df['price'])
except:
    print('Error with data.csv')
    sys.exit(1)

coef_X = X.max()
coef_Y = Y.max()
# scaling
X = X / coef_X
Y = Y / coef_Y

RMSE = [] # Root Mean Squared Error Tab
THETA = [] # Theta Tab

# learning...
for i in range(nb_iter):
    grad_b = (a * X + b - Y).mean()
    grad_a = ((a * X + b - Y) * X).mean()
    # save history
    if i < 10 or (i < 200 and i % 10 == 0) or i % 100 == 0:
        loss = ((a * X + b - Y) **2 ).mean()
        RMSE.append(loss**0.5 * coef_Y)
        THETA.append([i, a * coef_Y / coef_X, b * coef_Y])

    b -= learning_rate * grad_b
    a -= learning_rate * grad_a

loss = ((a * X + b - Y) **2 ).mean()
RMSE.append(loss**0.5 * coef_Y)
THETA.append([nb_iter, a * coef_Y / coef_X, b * coef_Y])

# sauvegarde des donnees dans result.csv
resfile = pd.DataFrame()
resfile['theta0'] = [b * coef_Y]
resfile['theta1'] = [a * coef_Y / coef_X]
resfile.to_csv('result.csv', index=False)

def visualize(X, Y, THETA, RMSE):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    main, err = axs
    fig.tight_layout(pad=3)
    itetab = []
    ploted = []
    for i in range(len(THETA)):
        ite, a, b = THETA[i]

        main.clear()
        main.set_ylabel('Price ($)')
        main.set_xlabel('Mileage (km)')
        main.set_title('Price / Mileage (iteration ' + str(ite) + ')')
        main.scatter(X, Y)
        main.plot(X, a*X+b, color='red')

        rmse = RMSE[i]
        itetab.append(ite)
        ploted.append(rmse)
        err.clear()
        err.set_title(f'Root Mean Squared Error / iterations    (current RMSE = {rmse:.5}$)')
        err.set_ylabel('RMSE ($)')
        err.set_xlabel('Iterations')
        err.plot(itetab, ploted)
        plt.pause(0.2)

    plt.show()

def bonus():
    print('MSE =', ((a * X + b - Y) **2 ).mean() * coef_Y ** 2)
    print('RMSE =', ((a * X + b - Y) **2 ).mean()**0.5 * coef_Y)
    # pour avoir une idee plus precise (qui ne depend plus de l'unité -> %), on peut calculer la RSE (Relative Squared Error)
    print('RSE =', ((a * X + b - Y) **2 ).mean() / ((Y - Y.mean())**2).mean())

    visualize(X * coef_X, Y * coef_Y, THETA, RMSE)

if len(sys.argv) > 1 and sys.argv[1] == 'bonus':
    bonus()
