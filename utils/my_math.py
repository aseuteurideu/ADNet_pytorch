import math
import numpy as np

# https://stackoverflow.com/a/41206290/3839572
def normal_round(n):
    if isinstance(n, (list, np.ndarray)):
        if type(n) is list:
            temp = np.array(n)
        else:
            temp = n

        for idx, value in np.ndenumerate(temp):
            if value - math.floor(value) < 0.5:
                temp[idx] = math.floor(value)
            temp[idx] = math.ceil(value)

        if type(n) is list:
            return list(temp)
        else:
            return temp

    else:
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
