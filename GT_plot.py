import h5py as h5
from matplotlib import pyplot as plt
import numpy as np


def generate_image_array(experiment_path):
    imgs = []
    with h5.File(experiment_path, 'r') as f:
        imgs.append(np.array(f['frame 5'], dtype=np.float64))
        imgs.append(np.array(f['frame 6'], dtype=np.float64))
        imgs.append(np.array(f['frame 7'], dtype=np.float64))
        imgs.append(np.array(f['frame 8'], dtype=np.float64))
        imgs.append(np.array(f['frame 9'], dtype=np.float64))
        '''imgs.append(np.array(f['frame 10'], dtype=np.float64))
        imgs.append(np.array(f['frame 11'], dtype=np.float64))
        imgs.append(np.array(f['frame 12'], dtype=np.float64))'''
    return imgs

def main():
    experiment_path = "/home/itsnas/ueuua/BA/dataset/visualize/ueuua_Cu_fans_100_40_100_HS_75_25_25.h5"
    imgs = generate_image_array(experiment_path)
    fig = plt.figure(figsize=(6, 6))
    columns = 2
    rows = 2
    i = 1
    # preds = [preds]
    # for entry in preds:
    fig.add_subplot(rows, columns, 1)
    plt.imshow(imgs[2]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 2)
    plt.imshow(imgs[3]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 3)
    plt.imshow(imgs[3]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 4)
    plt.imshow(imgs[4]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    '''fig.add_subplot(rows, columns, 5)
    plt.imshow(imgs[4]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 6)
    plt.imshow(imgs[5]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 7)
    plt.imshow(imgs[5]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 8)
    plt.imshow(imgs[6]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 9)
    plt.imshow(imgs[6]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 10)
    plt.imshow(imgs[7]*255, cmap='jet', vmin=30, vmax=170)
    
    plt.xticks([], [])
    plt.yticks([], [])'''
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    #plt.suptitle(f'Prediction of next {i} frames', y=0.65)
    plt.savefig('GT.png')
    
    fig = plt.figure(figsize=(6, 6))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(imgs[0]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 2)
    plt.imshow(imgs[1]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 3)
    plt.imshow(imgs[1]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 4)
    plt.imshow(imgs[2]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    '''fig.add_subplot(rows, columns, 5)
    plt.imshow(imgs[2]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 6)
    plt.imshow(imgs[3]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 7)
    plt.imshow(imgs[3]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 8)
    plt.imshow(imgs[4]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 9)
    plt.imshow(imgs[4]*255, cmap='jet', vmin=30, vmax=170)
    plt.xticks([], [])
    plt.yticks([], [])
    fig.add_subplot(rows, columns, 10)
    plt.imshow(imgs[5]*255, cmap='jet', vmin=30, vmax=170)
    
    plt.xticks([], [])
    plt.yticks([], [])'''
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)
    #plt.suptitle(f'Prediction of next {i} frames', y=0.65)
    plt.savefig('Input.png')


if __name__ == "__main__":
    main()
