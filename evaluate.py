#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

# from __future__ import print_function
# https://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works

"""
evaluate.py: script to evaluate tensorflow model
"""

__author__ = "Hua Zhao"

from src.glob import *
from src.utils import *
from src.utils_plot import _heatmap, _annotate_heatmap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import cv2


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
    """
    confusion matrix, TPR, PPV, sensitivity, precision, recall, etc,: 
    https://en.wikipedia.org/wiki/Confusion_matrix
    """
    
    if not os.path.isdir(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'])):
        os.makedirs(os.path.join(params['evaluate']['dir_prefix'], params['model']['name']))
        
    # transform to confusion matrix
    n_classes = len(class_names)
    counter = Counter(results)
    cmat = pd.DataFrame(index=list(range(len(labelmap))), columns=list(range(len(labelmap))))
    for y_true in range(n_classes):
        for y_pred in range(n_classes):
            cmat.loc[y_true, y_pred] = counter[(y_true, y_pred)]
    cmat_hnorm = (cmat.T / cmat.sum(axis=1)).T  # horizontally normalized for TPR/Sensitivity/Recall, TNR/Specificity
    cmat_vnorm = cmat / cmat.sum(axis=0)  # horizontally normalized for PPV/Precision, NPV
    def _sort(mat):
        mat.sort_index(inplace=True)
        mat = mat[sorted(mat.columns)]
        return mat
    cmat_hnorm = _sort(cmat_hnorm)
    cmat_vnorm = _sort(cmat_vnorm)
    # plot heatmap
    def _plot(df, fn):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(1, 1, 1)
        im, cbar = _heatmap(df.values.astype(float), class_names, class_names, ax=ax, cmap="Blues", cbarlabel=None)  # avoid type mismatch
        texts = _annotate_heatmap(im, valfmt="{x:.2f}")
        ax.set_title('Horizontally Normalized Confusion Matrix (TPR, TNR) of model lowest valid loss')
        ax.set_xlabel('Pred')
        ax.set_ylabel('True')
        fig.savefig(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], f'{fn}.png'))
        return
    _plot(cmat, 'confusion_matrix')
    _plot(cmat_hnorm, 'confusion_matrix_hnorm')
    _plot(cmat_vnorm, 'confusion_matrix_vnorm')
    return


def test():
    return


if __name__ == '__main__':
    test()
    pass