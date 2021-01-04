import math
import time

import matplotlib.pyplot as plt
import torch
from matplotlib import ticker


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def plot_loss_history(points, filepath=None, labels=None):
    labels = labels if labels is not None else ["Train"]
    plt.switch_backend('agg')
    num_series = 1 if type(points[0]) == float else len(points[0])
    assert num_series == len(labels)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    if num_series == 1:
        plt.plot(points)
    else:
        for i in range(num_series):
            ax.plot([point[i] for point in points])
    ax.legend(labels)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)


def compute_f1(y_pred: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.round(y_pred)
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return precision, recall, f1


def num_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)