import sys
import os.path
import csv


# def normalize(min: int, max: int) -> float:


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
            data: list[tuple[int, int]] = []
            for km, price in read_content:
                data.append((int(km), int(price)))
        print(data)
    except AssertionError as msg:
        print(msg)
    except ValueError:
        print("There is a problem in your data file.")

    # for km
    min_km = min(data)[0]
    max_km = max(data)[0]
    min_price = lambda x, y: min(x, y[0])
    max_price = max(data)[1]
    print(min_price, max_price)
    print(min(data))
    # min_km = min([(y, x) for km, price in data])[::-1]
    # key=lambda x:x[1]

    # print(f'{os.getlogin()}! How are you?')
