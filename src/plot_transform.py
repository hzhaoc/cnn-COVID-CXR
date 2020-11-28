#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
plot_feature.py: visualize model features from random samples to display ROI (regions of interest) the moodel focuses on
"""

__author__ = "Hua Zhao"

from src.glob import *
from matplotlib import pyplot as plt
from src.transform import *


def plot_example_transforms(example_num=5, save_path='./diag/', size=5):
    """
    visualize random original and transformed training or testing images
    """
    _processor = ImgPreprocessor(CLAHE=params['etl']['use_CLAHE'], 
                            crop_top=params['etl']['crop_top'], 
                            size=params['etl']['image_size'], 
                            clipLimit=params['etl']['CLAHE_clip_limit'],
                            tileGridSize=(params['etl']['CLAHE_tile_size'], params['etl']['CLAHE_tile_size'])
    )

    shutil.rmtree(save_path)
    os.makedirs(save_path)

    meta = pickle.load(open(os.path.join(SAVE_PATH,  'meta'), 'rb'))
    fns = np.random.choice(meta.img, 10)
    
    fig, axs = plt.subplots(len(fns), 2, constrained_layout=True)
    fig.set_size_inches(size*2, size*len(fns))

    for i, fn in enumerate(fns):
        if fn[-3:] == 'dcm':  # .dcm
            ds = dicom.dcmread(fn)
            img = ds.pixel_array
            img = cv2.merge((img,img,img))  # CXR images are exactly or almost gray scale (meaning 3 depths have very similar or same values); checked
        else:  # .png., .jpeg, .jpg
            img = cv2.imread(fn)

        img_pre = cv2.resize(img, (params['etl']['image_size'], params['etl']['image_size']))
        img_cur = _processor(img)

        axs[i, 0].imshow(img_pre)
        axs[i, 0].set_title(f"example {i+1} - original")
        
        axs[i, 1].imshow(img_cur)
        axs[i, 1].set_title(f"example {i+1} - transformed")

    title = f'images: original vs transformed'
    fig.suptitle(title, fontsize=16)
    plt.show()
    fig.savefig(os.path.join(save_path, f'transformation plot.png'))
    return