import sys
import os
import csv

def normalize(min: int, max: int) -> float:
    


if __name__ == "__main__":
    try:
        assert len(sys.argv) == 2, "You need to enter the path of your data file!"
        assert os.path.isfile(sys.argv[1]), "There is a problem with you file path..."
        with open(sys.argv[1], "r") as csvfile:
            read_content = csv.reader(csvfile, delimiter=",")
            next(read_content)
            data: list[dict[str, int]] = []
            for km, price in read_content:
                data.append({'km': int(km), 'price': int(price)})
        print(data)
    except AssertionError as msg:
        print(msg)
    except ValueError:
        print("There is a problem in your data file.")
    except PermissionError:
        print("I can't open your data file.")

    min_km = min(row['km'] for row in data)
    max_km = max(row['km'] for row in data)
    min_price = min(row['price'] for row in data)
    max_price = max(row['price'] for row in data)
    print(min_km, max_km)
    print(min_price, max_price)
    
    

    # print(f'{os.getlogin()}! How are you?')
