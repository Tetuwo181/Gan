# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def save_by_grid(img, file_path: str, show_raw: int, show_cal: int):
    fig, axs = plt.subplots(show_raw, show_cal)
    iteration = 0
    for i in range(show_raw):
        for j in range(show_cal):
            axs[i, j].imshow(img[iteration, :, :, :])
            axs[i, j].axis('off')
            iteration = iteration + 1
    fig.savefig(file_path)
