import sys
import os
import csv
import matplotlib.pyplot as plt


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
        new_km: float = value["km"] - min_km / gap_km
        normalized_km.append(new_km)
        new_price: float = value["price"] - min_price / gap_price
        normalized_price.append(new_price)
        # new_data.append({"km": new_km, "price": new_price})

    return normalized_km, normalized_price


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
        print(data)
        normalized_km, normalized_price = normalize(data)
        axes = plt.axes()
        axes.grid()
        plt.scatter(normalized_price, normalized_km)
        plt.show()

    except AssertionError as msg:
        print(msg)
    except ValueError:
        print("There is a problem in your data file.")
    except PermissionError:
        print("I can't open your data file.")

    # print(f'{os.getlogin()}! How are you?')
