from matplotlib import pyplot as plt
import numpy as np
import h5py as h5
import os


def evaluate_control(data_path, data):
    std_history = []
    hot_spot_history = []
    mean_history = []
    for f in data:
        print(f)
        img_path = os.path.join(data_path, f)
        with h5.File(img_path, 'r') as file:
            img = np.array(file['D2_dataNorm'], dtype=np.float64)
        std = np.std(img)
        mean = np.mean(img)
        hot_pixels = img - mean
        hot_score = np.sum(hot_pixels, where=hot_pixels > 0)
        std_history.append(std)
        hot_spot_history.append(hot_score)
        mean_history.append(mean)
    return std_history, hot_spot_history, mean_history

def plot(data_path):
    path_specialist = os.path.join(data_path, 'specialist')
    path_general = os.path.join(data_path, 'general')
    path_control = os.path.join(data_path, 'control')

    data_specialist = os.listdir(path_specialist)
    data_general = os.listdir(path_general)
    data_control = os.listdir(path_control)

    std_specialist, hot_spot_specialist, mean_specialist = evaluate_control(path_specialist, data_specialist)
    std_general, hot_spot_general, mean_general = evaluate_control(path_general, data_general)
    std_control, hot_spot_control, mean_control = evaluate_control(path_control, data_control)

    '''hot_spot_specialist = np.divide(hot_spot_specialist, hot_spot_specialist[0])
    hot_spot_general = np.divide(hot_spot_general, hot_spot_general[0])
    hot_spot_control = np.divide(hot_spot_control, hot_spot_control[0])

    mean_specialist = np.divide(mean_specialist, mean_specialist[0])
    mean_general = np.divide(mean_general, mean_general[0])
    mean_control = np.divide(mean_control, mean_control[0])

    std_specialist = np.divide(std_specialist, std_specialist[0])
    std_general = np.divide(std_general, std_general[0])
    std_control = np.divide(std_control, std_control[0])'''

    x = np.linspace(1, 11, num=11)

    plt.plot(x, hot_spot_specialist, color='r', marker='x', label='specialist')
    plt.plot(x, hot_spot_general, color='b', marker='o', label='general')
    plt.plot(x, hot_spot_control, color='g', marker='^', label='random')
    plt.title("Hot Spot Score Analysis")
    plt.legend()
    plt.show()

    plt.plot(x, mean_specialist, color='r', marker='x', label='specialist')
    plt.plot(x, mean_general, color='b', marker='o', label='general')
    plt.plot(x, mean_control, color='g', marker='^', label='random')
    plt.title("Mean Temperature")
    plt.legend()
    plt.show()

    plt.plot(x, std_specialist, color='r', marker='x', label='specialist')
    plt.plot(x, std_general, color='b', marker='o', label='general')
    plt.plot(x, std_control, color='g', marker='^', label='random')
    plt.title("STD Analysis")
    plt.legend()
    plt.show()


def main():
    data_path = 'H://public//images//GP//low_1'
    experiment_path = os.path.join(data_path, 'specialist')
    data = os.listdir(experiment_path)
    rows = 3
    columns = 11
    fig = plt.figure(figsize=(11, 3))
    for i, f in enumerate(data):
        print(f)
        img_path = os.path.join(experiment_path, f)
        with h5.File(img_path, 'r') as file:
            img = np.array(file['D2_dataNorm'], dtype=np.float64)

        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img*255, cmap='jet', vmin=30, vmax=170)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    data_path = 'H://public//images//GP//low_1'
    experiment_path = os.path.join(data_path, 'general')
    data = os.listdir(experiment_path)
    for i, f in enumerate(data):
        print(f)
        img_path = os.path.join(experiment_path, f)
        with h5.File(img_path, 'r') as file:
            img = np.array(file['D2_dataNorm'], dtype=np.float64)

        fig.add_subplot(rows, columns, i + 12)
        plt.imshow(img * 255, cmap='jet', vmin=30, vmax=170)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())


    data_path = 'H://public//images//GP//low_1'
    experiment_path = os.path.join(data_path, 'control')
    data = os.listdir(experiment_path)
    for i, f in enumerate(data):
        print(f)
        img_path = os.path.join(experiment_path, f)
        with h5.File(img_path, 'r') as file:
            img = np.array(file['D2_dataNorm'], dtype=np.float64)

        fig.add_subplot(rows, columns, i + 23)
        plt.imshow(img * 255, cmap='jet', vmin=30, vmax=170)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(top=0.7, bottom=0.3, hspace=0)
    plt.show()


def fan_loads():
    specialist = [0, 80, 149, 234, 229, 206, 205, 194, 197, 198, 192]
    general = [0, 80, 136, 176, 134, 150, 168, 155, 160, 95, 161]
    control = [0] + [90]*10
    x = np.linspace(1, 11, num=11)

    plt.step(x, specialist, color='r', marker='x', label='specialist')
    plt.step(x, general, color='b', marker='o', label='general')
    plt.step(x, control, color='g', marker='^', label='random')
    plt.title("Fan Load Analysis")
    plt.legend()
    plt.show()
    return control

if __name__ == '__main__':
    #plot('H://public//images//GP//hot_1')
    #main()
    fan_loads()


