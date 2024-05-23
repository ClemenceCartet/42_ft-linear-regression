"""This program will predict the price of a car for a given mileage"""

import json
import sys
import numpy as np

from utils import set_factors, normalize, denormalize


def model(
    factors: np.ndarray[np.float64, np.float64], thetas: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """Function that will calculate the estimated price of a car for a given mileage (factors)"""
    return factors.dot(thetas)


if __name__ == "__main__":
    try:
        with open("thetas.json", "r") as file:
            data = json.load(file)
            thetas = np.array(data)[:, np.newaxis]
        while True:
            val = input("Enter a mileage: ")
            if val == "exit":
                sys.exit()
            try:
                value = float(val)
                if value < 0:
                    print("You must enter a positve mileage.")
                    continue
                norm_value = normalize(value, 22899.0, 217101.0)
                m_factors = set_factors([norm_value])
                price = model(m_factors, thetas)
                denorm_price = denormalize(price.item(), 3650.0, 4640.0)
                print(f"The estimated price for your mileage is {denorm_price:.2f}")
            except ValueError:
                print("You must enter a valid mileage.")
                continue
    except Exception:
        print("There is no file. You must first train your model.")
