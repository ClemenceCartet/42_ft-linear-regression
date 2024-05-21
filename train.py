import sys
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def normalize(data: list[dict[str, int]]) -> tuple[list[float], list[float]]:
    """Normalize datas"""
    min_km = min(row["km"] for row in data)
    max_km = max(row["km"] for row in data)
    gap_km = max_km - min_km
    min_price = min(row["price"] for row in data)
    max_price = max(row["price"] for row in data)
    gap_price = max_price - min_price
    # print(min_km, max_km)
    # print(min_price, max_price)

    normalized_km: list[float] = []
    normalized_price: list[float] = []
    # new_data: list[dict[str, float]] = []
    for value in data:
        new_km: float = (value["km"] - min_km) / gap_km
        normalized_km.append(new_km)
        new_price: float = (value["price"] - min_price) / gap_price
        normalized_price.append(new_price)
        # new_data.append({"km": new_km, "price": new_price})

    return normalized_km, normalized_price


def model(
    factors: np.ndarray[np.float64, np.float64], thetas: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    """Model"""
    return factors.dot(thetas)


def cost_function(predictions, prices) -> tuple[float, np.ndarray[np.float64]]:
    """Sum of all errors divided by number of data"""
    return sum(np.power(np.subtract(predictions, prices), 2)) / (
        2 * len(prices)
    )


def find_gradients(predictions, prices, factors) -> tuple[float, float]:
    """Find gradients with partial differential of thetas"""
    factors_transposed = np.transpose(factors)
    gradients = factors_transposed.dot(np.subtract(predictions, prices))

    return gradients / len(prices)


def gradient_descent_algorithm(thetas, gradients, learning_rate: float):
    """Gradient Descent algorithm"""
    new_thetas = np.subtract(thetas, np.multiply(gradients, learning_rate))
    print(new_thetas)


if __name__ == "__main__":
    try:
        assert (
            len(sys.argv) == 2
        ), "You need to enter the path of your data file!"
        assert os.path.isfile(
            sys.argv[1]
        ), "There is a problem with you file path..."
        with open(sys.argv[1], "r") as csvfile:
            read_content = csv.reader(csvfile, delimiter=",")
            next(read_content)
            data: list[dict[str, int]] = []
            for km, price in read_content:
                data.append({"km": int(km), "price": int(price)})
        normalized_km, normalized_price = normalize(data)

        # thetas: np.ndarray[np.float64] = np.random.rand(2, 1)
        thetas = np.zeros((2, 1))
        a = np.array(normalized_km)[:, np.newaxis]
        b = np.ones((len(normalized_km), 1))
        factors: np.ndarray[np.float64, np.float64] = np.hstack((a, b))
        predictions = model(factors, thetas)

        prices = np.array(normalized_price)[:, np.newaxis]
        cost = cost_function(predictions, prices)
        print(f"Cost: {cost}")

        gradients = find_gradients(predictions, prices, factors)
        print(f"Gradients: {gradients}")

        gradient_descent_algorithm(thetas, gradients, 2.0)

        plt.rcParams["figure.figsize"] = [12, 6]
        fig, axes = plt.subplots(nrows=1, ncols=2)
        # axes = plt.axes()
        # axes.grid()
        axes[0].grid()
        axes[0].scatter(normalized_price, normalized_km, color="blue")
        axes[0].plot(predictions, normalized_km, color="red")
        axes[0].set_xlabel("Price")
        axes[0].set_ylabel("Mileage")
        axes[0].set_title("Predictions with datas")
        axes[0].legend(["Datas", "Predictions"])  # loc="lower right"
        plt.show()

    except AssertionError as msg:
        print(msg)
    except ValueError:
        print("There is a problem in your data file.")
    except PermissionError:
        print("I can't open your data file.")
