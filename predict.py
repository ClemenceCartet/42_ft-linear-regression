"""This program will predict the price of a car for a given mileage"""

import json
import sys
import numpy as np

from utils import set_factors, normalize, denormalize


def model(
    factors: np.ndarray[np.float64, np.float64], thetas: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """Function that will calculate the estimated price
    of a car for a given mileage (factors)"""
    return factors.dot(thetas)


if __name__ == "__main__":
    try:
        with open("thetas.json", "rb") as file:
            data = json.load(file)
            thetas = np.array(data["thetas"], dtype=float)[:, np.newaxis]
            min_mileage, range_mileage = (
                float(data["min_mileage"]),
                float(data["range_mileage"]),
            )
            min_price, range_price = float(data["min_price"]), float(
                data["range_price"]
            )
    except OSError:
        print("There is no file. You must first train your model.")
        thetas = np.zeros((2, 1))
        min_mileage, range_mileage = 0.0, 1.0
        min_price, range_price = 0.0, 1.0
    except ValueError:
        print("There is a problem in you thetas file.")
        sys.exit()

    while True:
        val = input("Enter a mileage: ")
        if val == "exit":
            sys.exit()
        try:
            value = float(val)
            if value < 0:
                print("You must enter a positve mileage.")
                continue
            norm_value = normalize(value, min_mileage, range_mileage)
            m_factors = set_factors([norm_value])
            price = model(m_factors, thetas)
            denorm_price = denormalize(price.item(), min_price, range_price)
            print(
                f"The estimated price for your mileage is \
{denorm_price:.2f}"
            )
        except ValueError:
            print("You must enter a valid mileage.")
            continue
