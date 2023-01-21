import numpy as np
import matplotlib.pyplot as plt
import torch


def imshow(image, ax=None, title=None, normalize=False):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()

def plot_wrong_pred_images(val_output, data, label):
    pred_labels = []
    for pred in val_output:
        pred_labels.append(int(torch.argmax(pred)))
    diff = torch.sub(torch.tensor(pred_labels), label)
    for ind, d in enumerate(diff):
        if d == -1:
            title = "cat" if pred_labels[ind] == 0 else "dog"
            imshow(data[ind], title=title)
