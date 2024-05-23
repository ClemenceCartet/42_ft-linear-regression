import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Any

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
    axes,
    factors: np.ndarray[np.float64, np.float64],
    thetas: np.ndarray[np.float64],
    prices: np.ndarray[np.float64],
    learning_rate: float,
) -> np.ndarray[np.float64]:
    """Training"""
    cost_history: list[float] = []
    thetas_history: list[np.ndarray[np.float64]] = []

    for i in range(200):
        # print(i)
        thetas_history.append(thetas)
        m_predictions = model(factors, thetas)
        cost_history.append(cost_function(m_predictions, prices))
        last_two = cost_history[-2:]
        # if len(last_two) > 1 and last_two[0] - last_two[1] < 10**-6:
        #     break
        gradients = find_gradients(m_predictions, prices, factors)
        thetas = gradient_descent_algorithm(thetas, gradients, learning_rate)

    axes[1].grid()
    axes[1].plot(list(range(i + 1)), cost_history, color="red")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].set_title("Cost function")

    axes[2].grid()
    axes[2].plot(
        list(range(i + 1)), [x[0] for x in thetas_history], color="red"
    )
    axes[2].plot(
        list(range(i + 1)), [x[1] for x in thetas_history], color="blue"
    )
    axes[2].set_xlabel("Iterations")
    axes[0].legend(["Theta1", "Theta0"])

    return m_predictions


if __name__ == "__main__":
    try:
        plt.rcParams["figure.figsize"] = [18, 6]
        fig, axes = plt.subplots(nrows=1, ncols=3)

        mileages, prices = get_datas("data.csv")
        min_mileage, gap_mileage = find_scale(mileages)
        min_price, gap_price = find_scale(prices)
        norm_mileages = [
            normalize(m, min_mileage, gap_mileage) for m in mileages
        ]
        norm_prices = [normalize(n, min_price, gap_price) for n in prices]

        m_factors = set_factors(norm_mileages)
        m_thetas = np.zeros((2, 1))
        m_norm_prices = np.array(norm_prices)[:, np.newaxis]

        m_predictions = training(axes, m_factors, m_thetas, m_norm_prices, 1)

        axes[0].grid()
        axes[0].scatter(norm_mileages, norm_prices, color="blue")
        axes[0].plot(norm_mileages, m_predictions, color="red")
        axes[0].set_xlabel("Mileage")
        axes[0].set_ylabel("Price")
        axes[0].set_title("Predictions with norm datas")
        axes[0].legend(["Datas", "Predictions"])  # loc="lower right"

        plt.show()

    except AssertionError as msg:
        print(msg)
    except PermissionError:
        print("I can't open your data file.")
    except ValueError:
        print("There is a problem in your data file.")
