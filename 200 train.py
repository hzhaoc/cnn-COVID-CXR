#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

# from __future__ import print_function
# https://stackoverflow.com/questions/7075082/what-is-future-in-python-used-for-and-how-when-to-use-it-and-how-it-works

"""
train.py: script to train COVID-net model
          original: https://github.com/lindawangg/COVID-Net
"""

__author__ = "Hua Zhao"

from evaluate import *
from src.models import *
from src.transform import Augmentator


def main():
    if params['model']['tool'] == 'pytorch':
        torch_train()
    elif (params['model']['tool'] == 'tensorflow') or (params['model']['tool'] == 'tf'):
        print('starting tensorflow model..')
        params['model']['tool'] = 'tensorflow'
        tf_train()
    else:
        raise ValueError(f"Invalid model architect tool {params['model']['tool']}")
    return


def tf_train():
    from src.label_balancer import TFBalancedCovidBatch  # include tensorflow
    
    # --------------------------------------------------------------------------------------------------------------------
    # DEPRECATED since loading large data into cache is too computing expensive
    # train_data / test_data structure: 
    #                        {'covid': {'data': list of size*size*3 numpy array, 'label': corresponding list of integer},
    #                        '!covid': {'data': list of size*size*3 numpy array, 'label': corresponding list of integer}}
    # check ./src/etl.py or ./etl.html for brief data description
    # train_data = pickle.load(open(os.path.join(params['data']['data_path'], params['data']['trainfile']), 'rb'))
    # test_data  = pickle.load(open(os.path.join(SAVE_PATH,  'test.data'), 'rb'))
    # --------------------------------------------------------------------------------------------------------------------
    
    # meta
    meta = pickle.load(open(os.path.join(SAVE_PATH,  'meta'), 'rb'))

    # batch generator
    batch_generator = BalancedCovidBatch(
                                is_training=True,
                                batch_size=params['train']['batch_size'],
                                batch_weight_covid=params['train']['sample_weight_covid'],
                                class_weights=params['train']['loss_weights']
    )
    # tensorflow session
    with tf.Session() as sess:
        tf.get_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(params['model']['tensorflow']['weightspath'], params['model']['tensorflow']['metaname']))
        graph = tf.get_default_graph()
        image_tensor = graph.get_tensor_by_name(params['model']['tensorflow']['in_tensorname'])
        labels_tensor = graph.get_tensor_by_name(params['model']['tensorflow']['label_tensorname'])
        sample_weights = graph.get_tensor_by_name(params['model']['tensorflow']['weights_tensorname'])
        pred_tensor = graph.get_tensor_by_name(params['model']['tensorflow']['logit_tensorname'])
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
        # saver.restore(sess, params['model']['tensorflow']['ckptname'])  # absolte path
        saver.restore(sess, os.path.join(params['model']['tensorflow']['weightspath'], params['model']['tensorflow']['ckptname']))  # relative path
        # saver.restore(sess, tf.train.latest_checkpoint(params['model']['weightspath']))

        # save base model
        # saver.save(sess, os.path.join(runPath, 'model'))
        # print('Saved baseline checkpoint')
        # print('Baseline eval:')
        # eval(sess, graph, testfiles, os.path.join(args.datadir,'test'), args.in_tensorname, args.out_tensorname, args.input_size)

        # Training
        n_batch = len(batch_generator)
        progbar = tf.keras.utils.Progbar(n_batch)
        if os.path.isfile(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'TPRs')):  # keep previous learning curves
            TPRs = pickle.load(open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'TPRs'), 'rb'))
            PPVs = pickle.load(open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'PPVs'), 'rb'))
            last_epoch = len(TPRs[list(TPRs.keys())[0]])

        else:  # start learning curves from refresh
            TPRs, PPVs = defaultdict(list), defaultdict(list)
            last_epoch = 0
        
        for epoch in range(last_epoch, last_epoch + params['train']['epochs']):
            for i in range(n_batch):
                _0 = time.time()
                batch_x, batch_y, weights = next(batch_generator)
                sess.run(train_op, feed_dict={image_tensor: batch_x, labels_tensor: batch_y, sample_weights: weights})
                progbar.update(i+1)
                _1 = time.time()
                print(f" - batch loss optimizing take {int(_1-_0)}s")

            # pred = sess.run(pred_tensor, feed_dict={image_tensor: batch_x})
            # loss = sess.run(loss, feed_dict={pred_tensor: pred, labels_tensor: batch_y, sample_weights: weights})
            # train_losses.append(loss)
            tpr, ppv = tf_evaluate(sess, graph, meta[meta.train!=1].copy(deep=True), 
                                params['model']['tensorflow']['in_tensorname'], params['model']['tensorflow']['logit_tensorname'], epoch)
            saver.save(sess, os.path.join('./model/', params['model']['name'], params['model']['name']), global_step=epoch+1, write_meta_graph=False)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))
            
            for label, i in labelmap.items():
                TPRs[label].append(tpr[i])  # this order is correct since sklearn.metrics.confusion_matrix is ordered by index that's the same index mapped from label
                PPVs[label].append(ppv[i])  # same
            
            if (i+1) % params['train']['display_step'] == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'COVID TPR {}\t'
                      'COVID PPV {}\t'.format(epoch, 
                                                  i, 
                                                  n_batch, 
                                                  tpr[labelmap['covid']], 
                                                  ppv[labelmap['covid']]))
        
        # plot and save learning curves
        tf_plot_learning_curves(TPRs, PPVs)

        with open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'TPRs'), 'wb') as pickle_file:
            pickle.dump(TPRs, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'PPVs'), 'wb') as pickle_file:
            pickle.dump(PPVs, pickle_file, pickle.HIGHEST_PROTOCOL)
        
    print("training finished.")


def torch_train():
    import torch
    import torchvision as tv
    from torchvision import datasets
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from src.label_balancer import pytorch_balanced_covid_samples as balancer

    # dirs
    if not os.path.isdir(os.path.join('./model/', params['model']['name'])):
        os.makedirs(os.path.join('./model/', params['model']['name']))
    if not os.path.isdir(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'])):
        os.makedirs(os.path.join(params['evaluate']['dir_prefix'], params['model']['name']))

    # train data
    augmentator = Augmentator(in_channel=params['model']['torch']['in_channel'])
    train_ds = datasets.ImageFolder(root='./data/train', transform=augmentator.pytorch_aumentator)
    _, _, sample_weights = balancer(train_ds.imgs, len(train_ds.classes), train_ds.class_to_idx, params['train']['sample_weight_covid'])  # train data -> balance sample classes
    sample_weights = torch.DoubleTensor(sample_weights)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=params['train']['batch_size'], sampler=train_sampler, num_workers=1, pin_memory=True)
    # test data
    test_ds = datasets.ImageFolder(root='./data/test', transform=augmentator.pytorch_aumentator)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=params['train']['batch_size'], shuffle=True, num_workers=1)
    
    # model setup
    arch = params['model']['architect']
    if params['model']['torch']['continue_learning']:  # continue learning from self-trained model at last point, load saved model first
        print('continue learning..')
        try:
            model = torch.load(os.path.join('./model/', params['model']['name'], params['model']['name']+'.last.pth'))
            train_losses = pickle.load(open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'train.losses'), 'rb'))
            valid_losses = pickle.load(open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'valid.losses'), 'rb'))
        except ValueError:
            print('are the torch model, train.losses, valid_losses all cached?')
        best_val_loss = min(valid_losses)
        last_epoch = len(valid_losses)
    else:  # start from a new model, wheter it's transfer learning or not
        print('transfer or fresh learning..')
        model = pytorch_model(architect='resnet18', pretrained=params['model']['torch']['transfer_learning'])
        last_epoch, train_losses, valid_losses, best_val_loss = 0, [], [], np.inf

    # traing, evaluate setup
    loss_weights = torch.FloatTensor([params['train']['loss_weights'][labelmap_inv[index]] for index in range(len(labelmap_inv))])
    criterion = nn.CrossEntropyLoss(weight=loss_weights)  # for weighted loss, see https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=params['train']['learning_rate'])
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=params['train']['lr_decay']['step_size'], gamma=params['train']['lr_decay']['gamma'])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # cuda not working well in my local machine
    model.to(device)
    criterion.to(device)
    
    isupdated = 0
    print('preloaded curve lentgh: ', len(train_losses)) 
    for epoch in range(last_epoch, last_epoch + params['train']['epochs']):
        train_loss, train_accuracy = _torch_train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = torch_evaluate(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_val_loss:  # let's keep the model that has the best loss, but you can also use another metric.
            isupdated, best_val_loss = 1, valid_loss
            torch.save(model, os.path.join('./model/', params['model']['name'], params['model']['name']+'.best.pth'))
    
    torch.save(model, os.path.join('./model/', params['model']['name'], params['model']['name']+'.last.pth'))
    print('aftertraining curve lentgh: ', len(train_losses))
    torch_plot_learning_curves(train_losses, valid_losses)

    if isupdated:  # if best model is updated
        best_model = torch.load(os.path.join('./model/', params['model']['name'], params['model']['name']+'.best.pth'))
        test_loss, test_accuracy, test_results = torch_evaluate(best_model, device, test_loader, criterion)
        torch_plot_confusion_matrix(test_results, test_ds.classes)

    with open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'train.losses'), 'wb') as pickle_file:
        pickle.dump(train_losses, pickle_file, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(params['evaluate']['dir_prefix'], params['model']['name'], 'valid.losses'), 'wb') as pickle_file:
        pickle.dump(valid_losses, pickle_file, pickle.HIGHEST_PROTOCOL)


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
    pass
    
if __name__ == '__main__':
    main()
    pass