import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from predict import model
from utils import get_datas, find_scale, normalize, set_factors


def cost_function(
    predictions: np.ndarray[np.float64], prices: np.ndarray[np.float64]
) -> float:
    """Sum of all errors divided by number of data"""
    return sum(np.power(np.subtract(predictions, prices), 2)) / (
        2 * len(prices)
    )


def find_gradients(
    predictions: np.ndarray[np.float64],
    prices: np.ndarray[np.float64],
    factors: np.ndarray[np.float64, np.float64],
) -> np.ndarray[np.float64]:
    """Find gradients with partial differential of thetas"""
    factors_transposed = np.transpose(factors)
    gradients = factors_transposed.dot(np.subtract(predictions, prices))

    return gradients / len(prices)


def gradient_descent_algorithm(
    thetas: np.ndarray[np.float64],
    gradients: np.ndarray[np.float64],
    learning_rate: float,
) -> np.ndarray[np.float64]:
    """Gradient Descent algorithm"""
    return np.subtract(thetas, np.multiply(gradients, learning_rate))


def training(
    axes, learning_rate: float, nb_iterations: int
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Training"""
    cost_history: list[float] = []
    factors = set_factors(norm_mileages)  # ndarray(m, 2)
    thetas = np.zeros((2, 1))  # ndarray(2, 1)
    prices = np.array(norm_prices)[:, np.newaxis]  # ndarray(m, 1)

    for i in range(nb_iterations):
        predictions = model(factors, thetas)
        cost_history.append(cost_function(predictions, prices))
        # last_two = cost_history[-2:]
        # if len(last_two) > 1 and last_two[0] - last_two[1] < 10**-6:
        #     break
        gradients = find_gradients(predictions, prices, factors)
        thetas = gradient_descent_algorithm(thetas, gradients, learning_rate)

    axes[1].plot(list(range(i + 1)), cost_history, color="red")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].set_title("Cost function")

    return predictions, thetas


if __name__ == "__main__":
    try:
        nb_iterations: int = 200
        learning_rate: float = 0.7
        if len(sys.argv) == 3:
            learning_rate = float(sys.argv[1])
            nb_iterations = int(sys.argv[2])
    except ValueError:
        print("There is a problem in your input.")
        sys.exit()

    try:
        mileages, prices = get_datas("data.csv")

        plt.rcParams["figure.figsize"] = [18, 6]
        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].grid()
        axes[1].grid()

        min_mileage, range_mileage = find_scale(mileages)
        min_price, range_price = find_scale(prices)
        norm_mileages = [
            normalize(m, min_mileage, range_mileage) for m in mileages
        ]
        norm_prices = [normalize(n, min_price, range_price) for n in prices]

        predictions, thetas = training(axes, learning_rate, nb_iterations)

        with open("thetas.json", "w") as file:
            json.dump(
                {
                    "thetas": [thetas[0].item(), thetas[1].item()],
                    "min_mileage": min_mileage,
                    "range_mileage": range_mileage,
                    "min_price": min_price,
                    "range_price": range_price,
                },
                file,
            )

        axes[0].scatter(norm_mileages, norm_prices, color="blue")
        axes[0].plot(norm_mileages, predictions, color="red")
        axes[0].set_xlabel("Mileage")
        axes[0].set_ylabel("Price")
        axes[0].set_title("Predictions with norm datas")
        axes[0].legend(["Datas", "Predictions"])

        plt.show()

    except AssertionError as msg:
        print(msg)
    except PermissionError:
        print("I can't open your data file.")
    except ValueError:
        print("There is a problem in your data file.")
