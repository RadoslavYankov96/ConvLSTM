import numpy as np
import os


def func(placeholder):
    files = os.listdir("fan_settings//")
    if files:
        f = files[-1]
        file_path = os.path.join("fan_settings", f)
        fan_settings = np.load(file_path)
        print(fan_settings)
        return fan_settings[0], fan_settings[1], fan_settings[2]
    else:
        return 0, 0, 0


if __name__ == "__main__":
    a, b, c = func([20, 40, 20])
