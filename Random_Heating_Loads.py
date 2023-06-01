import random
import os
import numpy as np


def generate_random_loads(list_of_loads):
    loads = random.choices(list_of_loads, k=3)
    return loads


def func(placeholder):
    path = "C://Users//ueuua//Desktop//Rado_BA_scripts//fan_settings//"
    files = os.listdir(path)
    if files:
        f = files[-1]
        file_path = os.path.join(path, f)
        fan_settings = np.load(file_path)
        print(fan_settings)
        return [fan_settings[0], fan_settings[1], fan_settings[2]]
        #return [0, 0, 0]
    else:
        return[30, 20, 30]


if __name__ == "__main__":
    func([0, 0, 0])
