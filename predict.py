import pandas as pd
import sys

def estimate_price(theta0, theta1, km):
    return theta0 + theta1 * km

# Recuperation des donnees depuis result.csv
try:
    resfile = pd.read_csv('result.csv')
except:
    print('Error: result.csv not found')
    sys.exit(1)

theta0, theta1 = float(resfile['theta0']), float(resfile['theta1'])
try:
    km = int(input('Mileage: '))
except:
    print('Error: mileage must be an integer')
    sys.exit(1)

print('Estimated price:', estimate_price(theta0, theta1, km))
