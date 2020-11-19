#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

# from __future__ import print_function
# https://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works

"""
train.py: script to train COVID-net model
          original: https://github.com/lindawangg/COVID-Net
"""

__author__ = "Hua Zhao"

from src.etl import *  # to load params
from src.utils import *
from evaluate import *
from collections import defaultdict
import time


def main():
    if params['model']['tool'] == 'torch':
        torch_train()
    elif (params['model']['tool'] == 'tensorflow') or (params['model']['tool'] == 'tf'):
        tf_train()
    else:
        raise ValueError(f"Invalid model architect tool {params['model']['tool']}")
    return


def tf_train():
    # import tensorflow.compat.v1 as tf  # version 2.x
    import tensorflow as tf  # version 1.x
    """
    NOTICE: 
    tensorflow default builds DO NOT include CPU instructions that fasten matrix computation including avx, avx2, etc,.
    see:
    (https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u)
    to solve this, download tailored wheel from:
    (https://github.com/fo40225/tensorflow-windows-wheel/tree/master/2.1.0/py37/CPU%2BGPU/cuda102cudnn76avx2)
    then isntall the package by
    'pip install --ignore-installed --upgrade /path/target.whl'
    """
    from src.batch_generator import BalancedCovidBatch
    
    # --------------------------------------------------------------------------------------------------------------------
    # DEPRECATED since loading large data into cache is too computing expensive
    # train_data / test_data structure: 
    #                        {'covid': {'data': list of size*size*3 numpy array, 'label': corresponding list of integer},
    #                        '!covid': {'data': list of size*size*3 numpy array, 'label': corresponding list of integer}}
    # check ./src/etl.py or ./etl.html for brief data description
    # print('reading data..')
    # train_data = pickle.load(open(os.path.join(params['data']['data_path'], params['data']['trainfile']), 'rb'))
    # test_data  = pickle.load(open(os.path.join(SAVE_PATH,  'test.data'), 'rb'))
    # --------------------------------------------------------------------------------------------------------------------
    
    # meta
    meta = pickle.load(open(os.path.join(SAVE_PATH,  'meta'), 'rb'))

    # batch generator
    batch_generator = BalancedCovidBatch(
                                is_training=True,
                                batch_size=params['train']['batch_size'],
                                batch_weight_covid=params['train']['batch_weight_covid'],
                                class_weights=params['train']['class_weights']
    )
    # tensorflow session
    with tf.Session() as sess:
        tf.get_default_graph()
        print('loading pretrained model..')
        saver = tf.train.import_meta_graph(os.path.join(params['model']['weightspath'], params['model']['metaname']))
        graph = tf.get_default_graph()
        image_tensor = graph.get_tensor_by_name(params['train']['in_tensorname'])
        labels_tensor = graph.get_tensor_by_name(params['train']['label_tensorname'])
        sample_weights = graph.get_tensor_by_name(params['train']['weights_tensorname'])
        pred_tensor = graph.get_tensor_by_name(params['train']['logit_tensorname'])
        # loss expects unscaled logits since it performs a softmax on logits internally for efficiency

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_tensor, labels=labels_tensor)*sample_weights)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['train']['learning_rate'])
        train_op = optimizer.minimize(loss)

        # Initialize the variables
        init = tf.global_variables_initializer()

        # Run the initializer
        sess.run(init)

        # load weights
        print('loading checkpoints..')
        saver.restore(sess, params['model']['ckptname'])  # absolute path
        # saver.restore(sess, os.path.join(params['model']['weightspath'], params['model']['ckptname']))  # relative path
        # saver.restore(sess, tf.train.latest_checkpoint(params['model']['weightspath']))

        # save base model
        # saver.save(sess, os.path.join(runPath, 'model'))
        # print('Saved baseline checkpoint')
        # print('Baseline eval:')
        # eval(sess, graph, testfiles, os.path.join(args.datadir,'test'), args.in_tensorname, args.out_tensorname, args.input_size)

        # Training
        print('Training started')
        n_batch = len(batch_generator)
        progbar = tf.keras.utils.Progbar(n_batch)
        TPRs, PPVs = defaultdict(list) , defaultdict(list) 
        
        for epoch in range(params['train']['epochs']):
            for i in range(n_batch):
                _0 = time.time()
                batch_x, batch_y, weights = next(batch_generator)
                sess.run(train_op, feed_dict={image_tensor: batch_x, labels_tensor: batch_y, sample_weights: weights})
                progbar.update(i+1)
                _1 = time.time()
                print(f"batch loss optimizing take {int(_1-_0)}s")

            print('calculating tpr, ppv from epoch..')
            # pred = sess.run(pred_tensor, feed_dict={image_tensor: batch_x})
            # loss = sess.run(loss, feed_dict={pred_tensor: pred, labels_tensor: batch_y, sample_weights: weights})
            # train_losses.append(loss)
            tpr, ppv = evaluate(sess, graph, meta[meta.train!=1].copy(deep=True), 
                                params['train']['in_tensorname'], params['train']['logit_tensorname'], epoch)
            for label, i in params['train']['labelmap'].items():
                TPRs[label].append(tpr[i])  # this order is correct since sklearn.metrics.confusion_matrix is ordered by label integers
                PPVs[label].append(ppv[i])  # same
            
            if (i+1) % params['train']['display_step'] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'COVID TPR {:.4f}\t'
                      'COVID PPV {:.4f}\t'.format(epoch, 
                                                  i, 
                                                  n_batch, 
                                                  tpr[params['train']['labelmap']['covid']], 
                                                  ppv[params['train']['labelmap']['covid']]))
                # eval(sess, graph, testfiles, os.path.join(args.datadir,'test'), args.in_tensorname, args.out_tensorname, args.input_size)
                # saver.save(sess, os.path.join(runPath, 'model'), global_step=epoch+1, write_meta_graph=False)
                # print('Saving checkpoint at epoch {}'.format(epoch + 1))
        
        # plot and save learning curves
        plot_learning_curves(TPRs, PPVs)
        
    print("training finished.")


def torch_train():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchvision import datasets
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets
    from torch.optim import lr_scheduler

    # data
    # pytorch augumentation, no need to use transforms.Normalize for TResNet, 
    # see https://github.com/mrT23/TResNet/issues/5#issuecomment-608440989
    _pytorch_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            torchvision.transforms.ColorJitter(hue=.1, saturation=.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor()
        ])
    # train data
    train_ds = datasets.ImageFolder(root='./data/train', transform=_pytorch_transform)
    # train data -> balance sample classes
    train_weights = torch_make_weights_for_balanced_classes(train_ds.imgs, len(train_ds.classes))
    train_weights = torch.DoubleTensor(train_weights)                                       
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['train']['batch_size'],
                                               sampler = train_sampler, num_workers=1, pin_memory=True)
    # test data
    test_ds = datasets.ImageFolder(root='./data/test', transform=_pytorch_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=params['train']['batch_size'], shuffle=True, num_workers=1)
    
    # model
    model = torchvision.models.resnet50(pretrained=True)  # ResNet-50
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(train_ds.classes))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=params['train']['learning_rate'])
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=params['train']['lr_decay']['step_size'], 
                                           gamma=params['train']['lr_decay']['gamma'])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    criterion.to(device)
    
    train_losses, valid_losses = [], []
    best_val_los = 100
    for epoch in range(params['train']['epochs']):
        train_loss, train_accuracy = _torch_train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = torch_evaluate(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        is_best = valid_loss < best_val_los
        if is_best:
            # best_val_acc = valid_accuracy
            best_val_los = valid_loss
            torch.save(model, os.path.join('./model/', params['model']['name']+'.pth'))
            
    torch_plot_learning_curves(train_losses, valid_losses)

    best_model = torch.load(os.path.join('./model/', params['model']['name']+'.pth'))
    test_loss, test_accuracy, test_results = torch_evaluate(best_model, device, test_loader, criterion)

    torch_plot_confusion_matrix(test_results, test_ds.classes)

    with open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'train.losses'), 'wb') as pickle_file:
        pickle.dump(train.losses, pickle_file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'valid.losses'), 'wb') as pickle_file:
        pickle.dump(valid.losses, pickle_file, pickle.HIGHEST_PROTOCOL)

def _torch_train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        # print('output size: ', output.size(), 'target size: ', target.size())
        target = target.long()  # for case: criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accuracy))
    return losses.avg, accuracy.avg


def test():
    # tensorflow session
    with tf.Session() as sess:
        tf.get_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(params['model']['weightspath'], params['model']['metaname']))
        graph = tf.get_default_graph()
        for op in graph.get_operations():
            if '/Softmax' in op.name:
                print(op.name)
    
    
if __name__ == '__main__':
    main()
    pass