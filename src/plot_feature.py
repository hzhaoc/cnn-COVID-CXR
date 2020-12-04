#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
plot_feature.py: visualize model features from random samples to display ROI (regions of interest) the moodel focuses on
"""

__author__ = "Hua Zhao"

from src import *
import torch
from gradcam.gradcam import *  # https://github.com/jacobgil/pytorch-grad-cam
from matplotlib import pyplot as plt


def plot_features_from_random_example(example_num=1, 
                                    model_tool='pytorch', 
                                    architect=None, 
                                    model_name=None, 
                                    save_path='./output/', 
                                    img_size_inch=6,
                                    in_channel=3,
                                    img_size=224,
):
    """
    # visualize model features from random samples to display ROI (regions of interest) the moodel focuses on
    ---------------------------
    models currenctly supported:
        - pytorch
            - ResNet-50
            - ResNet-18
            - VGG-19
            - VGG-11
        - tensorflow
            - N/A
    ---------------------------
    """
    if os.path.isdir(os.path.join(save_path, 'feature')):
        shutil.rmtree(os.path.join(save_path, 'feature'))
    os.makedirs(os.path.join(save_path, 'feature'))

    if model_tool == 'pytorch':
        plot_example_torch_features(example_num=example_num, 
                                    model_name=model_name, 
                                    save_path=save_path,
                                    architect=architect,
                                    img_size_inch=img_size_inch,
                                    in_channel=in_channel,
                                    img_size=224,
                                    )
    elif (model_tool == 'tensorflow') or (model_tool == 'tf'):
        plot_example_tf_features(example_num=example_num, 
                                 model_name=model_name, 
                                 save_path=save_path,
                                 architect=architect,
                                 img_size_inch=img_size_inch,
                                 in_channel=in_channel,
                                 img_size=224,
                                 )
    else:
        raise ValueError(f"Invalid model architect tool {model_too}")


def plot_example_torch_features(example_num=1, model_name='test', architect=None, save_path='./output/', img_size_inch=6, in_channel=3, img_size=224):
    """
    python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. 
    """
    
    model = torch.load(os.path.join('./model/', model_name, model_name+'.best.pth'))
    
    if 'resnet50' == architect.lower():
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=False)
    elif 'resnet18' == architect.lower():
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["1"], use_cuda=False)
    elif 'vgg19' == architect.lower():
        grad_cam = GradCam(model=model, feature_module=model.features, target_layer_names=["36"], use_cuda=False)
    elif 'vgg11' == architect.lower():
        grad_cam = GradCam(model=model, feature_module=model.features, target_layer_names=["20"], use_cuda=False)
    else:
        raise ValueError(f"Invalid model architect {architect}")
    
    meta = pickle.load(open(os.path.join(CACHE_PATH, 'meta', 'meta'), 'rb'))
    
    for example in range(example_num):
        fig, axs = plt.subplots(len(labelmap), 2, constrained_layout=True)
        fig.set_size_inches(img_size_inch*2, img_size_inch*len(labelmap))
        for i, (label, label_i) in enumerate(labelmap.items()):

            rand_idx = random.choice(meta[meta.label==label_i].index)

            img = cv2.imread(meta.loc[rand_idx, 'imgid'], 1)  # cv2.IMREAD_COLOR
            # img = cv2.imread(meta.loc[rand_idx, 'imgid'], cv2.IMREAD_GRAYSCALE)  # 0
            img = np.float32(cv2.resize(img, (img_size, img_size))) / 255
            
            input1 = preprocess_image(img)
            if in_channel == 1:
                input1 = input1[:, 0, :, :].reshape((1, 1, img_size, img_size))  # since source images are almost gray, scicing RGB is okay

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
            target_index = None
            mask = grad_cam(input1, target_index)

            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)

            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, 0].imshow(img_color)
            axs[i, 0].set_title(f"{label} - original")
            
            gray = cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)
            axs[i, 1].imshow(gray)
            axs[i, 1].set_title(f"{label} - model feature")
        title = f'feature example {example+1}'
        fig.suptitle(title, fontsize=16)
        plt.show()
        fig.savefig(os.path.join(save_path, 'feature', f'{model_name}.feature.{example+1}.png'))
    return


def plot_example_tf_features(example_num=1, model_name='test', architect=None, save_path='./output/', img_size_inch=6):
    # new_model = tf.saved_model.load_v2('./model/COVIDNet-CXR-Small/savedModel')
    # new_model = tf.keras.models.load_model('./model/COVIDNet-CXR-Large/savedModel')
    raise Exception('Feature visualization for COVID-Net not supported now. Unable to load target tensorflow saved model downloaded from source')
    return -1