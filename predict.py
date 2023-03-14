import pandas as pd

def estimate_price(theta0, theta1, km):
    return theta0 + theta1 * km

# Recuperation des donnees depuis result.csv
resfile = pd.read_csv('result.csv')
theta0, theta1 = float(resfile['theta0']), float(resfile['theta1'])

km = int(input('Mileage: '))
print('Estimated price:', estimate_price(theta0, theta1, km))
