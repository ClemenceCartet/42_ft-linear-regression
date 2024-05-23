"""This program will predict the price of a car for a given mileage"""

import numpy as np


def model(
    factors: np.ndarray[np.float64, np.float64], thetas: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """Function that will calculate the estimated price of a car for a given mileage (factors)"""
    return factors.dot(thetas)


if __name__ == "__main__":
    val = input("Enter a mileage: ")
    try:
        mileage = np.array(float(val))[:, np.newaxis]
        price = model(factors, thetas)
        print(f"The estimated price for your mileage is {price[0]}")
    except ValueError:
        print("There is a problem in your input.")
