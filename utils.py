"""Utils functions"""

import csv
import os

import numpy as np


def find_scale(datas: list[float]) -> tuple[float, float]:
    """Find min and max of datas, and return min and range"""
    minimum = min(datas)
    maximum = max(datas)
    data_range = maximum - minimum

    return minimum, data_range


def normalize(m: float, minimum: float, data_range: float) -> float:
    """Normalize data"""
    return (m - minimum) / data_range


def denormalize(n: float, minimum: float, data_range: float) -> float:
    """Denormalize data"""
    return n * data_range + minimum


def set_factors(mileage: list[float]) -> np.ndarray[np.float64, np.float64]:
    """Create a matrice with factors and bias"""
    a = np.array(mileage)[:, np.newaxis]
    b = np.ones((len(mileage), 1))

    return np.hstack((a, b))


def get_datas(file: str) -> tuple[list[float], list[float]]:
    """Check data file, and return two list of mileages and prices"""
    assert os.path.isfile(file), "There is a problem with you file path..."
    with open(file, "r") as csvfile:
        read_content = csv.reader(csvfile, delimiter=",")
        next(read_content)

        mileages: list[float] = []
        prices: list[float] = []
        for km, price in read_content:
            mileages.append(float(km))
            prices.append(float(price))

    return mileages, prices
