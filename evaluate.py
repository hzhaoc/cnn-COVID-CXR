#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

# from __future__ import print_function
# https://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works

"""
evaluate.py: script to evaluate tensorflow model
"""

__author__ = "Hua Zhao"

from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os
import cv2
from src.etl import *  # to load params, global 'variables'
import matplotlib.pyplot as plt
import matplotlib
from src.utils import *
from collections import Counter


@timmer(1, 0, 0)
def torch_evaluate(model, device, data_loader, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    results = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            target = target.long()  # for case: criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

    return losses.avg, accuracy.avg, results


@timmer(1, 0, 0)
def tf_evaluate(sess, graph, meta, input_tensor, output_tensor, epoch):
    input_tensor = graph.get_tensor_by_name(input_tensor)
    output_tensor = graph.get_tensor_by_name(output_tensor)
    y_true, y_pred = [], []
    n = len(meta)
    meta.reset_index(drop=True, inplace=True)
    print(f'evaluating {n} samples..')
    for _, sample in meta.iterrows():
        print("    progress: {0:.2f}%".format((_ + 1) * 100 / n), end="\r")
        x = cv2.imread(sample.imgid)
        x = x.astype('float32') / 255.0  # normalize
        y_true.append(sample.label)
        y_pred.append(np.array(sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(x, axis=0)})).argmax(axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # confusion matrix
    matrix = confusion_matrix(y_true, y_pred)  # ordered by label acsending
    matrix = matrix.astype('float')
    tprs = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    """
    print('\n', matrix)
    print('Sens/TPR Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(tprs[params['train']['labelmap']['normal']],
                                                                                 tprs[params['train']['labelmap']['pneumonia']],
                                                                                 tprs[params['train']['labelmap']['covid']]))
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[params['train']['labelmap']['normal']],
                                                                             ppvs[params['train']['labelmap']['pneumonia']],
                                                                             ppvs[params['train']['labelmap']['covid']]))
    """
    # plot and save normalized confusion matrix
    tf_plot_confusion_matrix(matrix, [labelmap_inv[x] for x in sorted(labelmap_inv)], epoch)

    return tprs, ppvs


def tf_plot_learning_curves(TPRs, PPVs):
    # TPR
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1)
    for label, series in TPRs.items():
        ax.plot(series, label=f'{label}')
    ax.legend(loc='upper left')
    ax.set_title('TPR / Sensitivity')
    ax.set_xlabel('epoch')
    ax.set_ylabel('rate')
    ax.grid()
    if not os.path.isdir(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'])):
        os.makedirs(os.path.join(params['evaluate']['dir_prefix'], params['model']['name']))
    fig.savefig(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'TPR.png'))
    # PPV
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1)
    for label, series in PPVs.items():
        ax.plot(series, label=f'{label}')
    ax.legend(loc='upper left')
    ax.set_title('PPV')
    ax.set_xlabel('epoch')
    ax.set_ylabel('rate')
    ax.grid()
    fig.savefig(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'PPV.png'))
    return


def tf_plot_confusion_matrix(matrix, class_names, epoch):
    # normalize confusion matrix (TPR)
    matrix = matrix / matrix.sum(axis=1).reshape([-1, 1])
    # plot heatmap
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    im, cbar = _heatmap(matrix, class_names, class_names, ax=ax, cmap="Blues", cbarlabel=None)
    texts = _annotate_heatmap(im, valfmt="{x:.2f}")
    ax.set_title('Normalized Confusion Matrix')
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    if not os.path.isdir(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'])):
        os.makedirs(os.path.join(params['evaluate']['dir_prefix'], params['model']['name']))
    fig.savefig(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], f'confusion_matrix_e{epoch}.png'))
    return


def torch_plot_learning_curves(train_losses, valid_losses):
    if not os.path.isdir(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'])):
        os.makedirs(os.path.join(params['evaluate']['dir_prefix'], params['model']['name']))
    
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 1, 1)
    ax.plot(train_losses, label='train loss')
    ax.plot(valid_losses, label='valid loss')
    ax.legend(loc='upper left')
    ax.set_title('loss curve')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    fig.savefig(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'losses.png'))
    pass


def torch_plot_confusion_matrix(results, class_names):
    if not os.path.isdir(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'])):
        os.makedirs(os.path.join(params['evaluate']['dir_prefix'], params['model']['name']))
        
    # transform to confusion matrix
    n_classes = len(class_names)
    counter = Counter(results)
    res = pd.DataFrame(index=list(range(len(labelmap))), columns=list(range(len(labelmap))))
    for y_true in range(n_classes):
        for y_pred in range(n_classes):
            res.loc[y_true, y_pred] = counter[(y_true, y_pred)]
    res = (res.T / res.sum(axis=1)).T
    res.sort_index(inplace=True)  # sort row
    res = res[sorted(res.columns)]  # sort column
    # print('testing normed confusion matrix:\n', res.values.astype(float))
    
    # plot heatmap
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(1, 1, 1)
    im, cbar = _heatmap(res.values.astype(float), class_names, class_names, ax=ax, cmap="Blues", cbarlabel=None)  # avoid type mismatch
    texts = _annotate_heatmap(im, valfmt="{x:.2f}")
    ax.set_title('Normalized Confusion Matrix')
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    fig.savefig(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'confusion_matrix_best_valid_loss.png'))
    pass


def _heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def _annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def test():
    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(params['model']['weightspath'], params['model']['metaname']))
    saver.restore(sess, params['model']['ckptname'])
    graph = tf.get_default_graph()
    
    META = pickle.load(open(os.path.join(SAVE_PATH,  'meta'), 'rb'))
    META = META[META.train!=1].iloc[:100, :]  # test samples
    evaluate(sess, graph, META.copy(deep=True), params['train']['in_tensorname'], params['train']['logit_tensorname'], 0)
    return


if __name__ == '__main__':
    test()
    pass