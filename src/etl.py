#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
etl.py: do ETL on source images
Source 0: https://github.com/ieee8023/covid-chestxray-dataset
Source 1: https://github.com/agchung/Figure1-COVID-chestxray-dataset
Source 2: https://github.com/agchung/Actualmed-COVID-chestxray-dataset
Source 3: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
Source 4: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
"""

__author__ = "Hua Zhao"

from src.glob import *
from src.utils import *
from src.transform import *


def preprocess(META):
    """
    preprocess and save source images into pipeline
    """
    for p in [TRAIN_PATH, TEST_PATH]:
        shutil.rmtree(p)
        os.makedirs(p)
        for label in labelmap.keys():
            if not os.path.isdir(os.path.join(p, label)):
                os.makedirs(os.path.join(p, label))

    _processor = ImgPreprocessor(CLAHE=params['etl']['use_CLAHE'], 
                            crop_top=params['etl']['crop_top'], 
                            size=params['etl']['image_size'], 
                            clipLimit=params['etl']['CLAHE_clip_limit'],
                            tileGridSize=(params['etl']['CLAHE_tile_size'], params['etl']['CLAHE_tile_size']),
                            use_seg=params['etl']['use_segmentation'],
    )
    print(_processor)
    n = len(META)
    for _, sample in META.iterrows():
        fn = sample.img
        if fn[-3:] == 'dcm':  # .dcm
            ds = dicom.dcmread(fn)
            img = ds.pixel_array
            img = cv2.merge((img,img,img))  # CXR images are exactly or almost gray scale (meaning 3 depths have very similar or same values); checked
        else:  # .png., .jpeg, .jpg
            img = cv2.imread(fn)
        img = _processor(img)
        cv2.imwrite(sample.imgid, img)
        print("progress: {0:.2f}%".format((_ + 1) * 100 / n), end="\r")
    return


def dataset_split(META):
    # train test split
    from sklearn.model_selection import train_test_split
    train0, test0 = train_test_split(META[META.label=='covid'], train_size=params['etl']['train_size'], random_state=params['etl']['split_rand'])
    train1, test1 = train_test_split(META[META.label=='pneumonia'], train_size=params['etl']['train_size'], random_state=params['etl']['split_rand'])
    train2, test2 = train_test_split(META[META.label=='normal'], train_size=params['etl']['train_size'], random_state=params['etl']['split_rand'])
    META_train = train0.append(train1).append(train2)
    META_train['train'] = 1
    META_test = test0.append(test1).append(test2)
    META_test['train'] = 0
    # together
    META = META_train.append(META_test).sort_index()
    META = META.reset_index(drop=True)
    META['label'] = META.label.map(labelmap)
    # to map images id from various sources
    META = META.reset_index().rename(columns={'index': 'imgid'})
    def _img(p):  # p -> patient (sample actually)
        _fn = f"{p.imgid}.{p.img.split('.')[-1]}" if p.src != 4 else f"{p.imgid}.png"
        _dir = TRAIN_PATH if p.train else TEST_PATH
        return os.path.join(_dir, labelmap_inv[p.label], _fn)
    META['imgid'] = META.apply(lambda p: _img(p), axis=1)
    return META


def etl():
    META_0 = etl_META_0()
    META_1 = etl_META_1()
    META_2 = etl_META_2()
    META_3 = etl_META_3()
    META_4 = etl_META_4()
    # together
    META = META_0.append(META_1).append(META_2).append(META_3).append(META_4)
    META.reset_index(drop=True, inplace=True)
    print('done')
    return META


def etl_META_0():
    # src 0
    META_0 = pd.read_csv(INPUT_PATH_0_META, nrows=None)
    global _src0_url
    _src0_url = META_0.url.to_list()  # for duplicates in src 3
    META_0 = META_0[META_0.view.isin(["PA", "AP", "AP Supine", "AP semi erect", "AP erect"])]  # filter views
    META_0 = META_0[['patientid', 'filename', 'finding']]
    META_0.dropna(axis=0, how='any', inplace=True)
    META_0.rename(columns={'patientid': 'patient', 'filename': 'img', 'finding': 'label'}, inplace=True)
    META_0['src'] = 0
    META_0['label'] = META_0.label.apply(src0_label)
    META_0['img'] = META_0.img.apply(lambda path: os.path.join(INPUT_PATH_0_IMG, path))
    META_0 = META_0[META_0.label!='other']
    return META_0


def etl_META_1():
    # src 1
    META_1 = pd.read_csv(INPUT_PATH_1_META, encoding='ISO-8859-1', nrows=None)
    META_1 = META_1[['patientid', 'finding']]
    META_1.dropna(axis=0, how='any', inplace=True)
    META_1.rename(columns={'patientid': 'patient', 'finding': 'label'}, inplace=True)
    META_1['img'] = META_1.patient.apply(src1_imgpath)
    META_1['img'] = META_1.img.apply(lambda path: os.path.join(INPUT_PATH_1_IMG, path))
    META_1['src'] = 1
    META_1['label'] = META_1.label.apply(src1_label)
    META_1 = META_1[META_1.label!='other']
    return META_1


def etl_META_2():
    # src 2
    META_2 = pd.read_csv(INPUT_PATH_2_META, nrows=None)
    META_2 = META_2[['patientid', 'finding', 'imagename']]
    META_2.dropna(axis=0, how='any', inplace=True)
    META_2.rename(columns={'patientid': 'patient', 'finding': 'label', 'imagename': 'img'}, inplace=True)
    META_2['label'] = META_2.label.apply(src2_label)
    META_2['img'] = META_2.img.apply(lambda path: os.path.join(INPUT_PATH_2_IMG, path))
    META_2 = META_2[META_2.label!='other']
    META_2['src'] = 2
    return META_2


def etl_META_3():
    df0 = src3_etl(INPUT_PATH_3_0_META, 'covid')
    df1 = src3_etl(INPUT_PATH_3_1_META, 'normal')
    df2 = src3_etl(INPUT_PATH_3_2_META, 'pneumonia')
    META_3 = df0.append(df1).append(df2)
    return META_3


def etl_META_4():
    # src 4
    META_4_1 = pd.read_csv(INPUT_PATH_4_META_1, nrows=None)
    META_4 = pd.read_csv(INPUT_PATH_4_META, nrows=None)
    # one img per patient, one patient per row
    META_4.drop_duplicates(keep='first', inplace=True)
    META_4_1.drop_duplicates(keep='first', inplace=True)
    META_4 = pd.merge(META_4, META_4_1, left_on='patientId', right_on='patientId', how='left')
    # according to src 4, kaggle file description, non-pneumonia has classes of normal and non-normal
    # for this project to classify [normal, pneumonia, COVID], we drop non-normal non-pneumonia
    _ = ((META_4.Target==1))|((META_4.Target==0)&(META_4['class']=='Normal'))
    META_4 = META_4[_][['patientId', 'Target']]
    META_4['img'] = META_4.patientId.apply(lambda p: os.path.join(INPUT_PATH_4_IMG, f'{p}.dcm'))
    META_4['Target'] = META_4.Target.apply(lambda t: 'normal' if not t else 'pneumonia')
    META_4.rename(columns={'Target': 'label', 'patientId': 'patient'}, inplace=True)
    META_4['src'] = 4
    return META_4


def src3_img(row):
    s = row['FILE NAME']
    if 'covid' in s.lower():
        fn = os.path.join(INPUT_PATH_3_0_IMG, f"{s}.{row.FORMAT.lower()}")
        _ = os.path.exists(fn)
    if 'normal' in s.lower():
        fn = os.path.join(INPUT_PATH_3_1_IMG, f"{'('.join(row['FILE NAME'].split('-'))}).{row.FORMAT.lower()}")
        _ = os.path.exists(fn)
    if 'pneumonia' in s.lower():
        fn = os.path.join(INPUT_PATH_3_2_IMG, f"{'('.join(row['FILE NAME'].split('-'))}).{row.FORMAT.lower()}")
        _ = os.path.exists(fn)
    return fn if _ else ' ('.join(fn.split('('))


def src3_etl(metapath, label):
    df = pd.read_csv(metapath)
    del df['SIZE']
    if label=='covid':
        # discard: # https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx.ipynb
        df = df[~df['FILE NAME'].isin(discard)]
        df = df[~df.URL.isin(_src0_url)]  # drop duplicates
    del df['URL']
    df.dropna(axis=0, how='any', inplace=True)
    df['img'] = df.apply(src3_img, axis=1)
    del df['FORMAT']
    df.rename(columns={'FILE NAME': 'patient'}, inplace=True)
    df['label'] = label
    df['src'] = 3
    return df


def src1_label(l):
    if not l:
        return 'other'
    l = l.lower()
    if 'pneumonia' in l:
        return 'pneumonia'
    elif 'covid' in l:
        return 'covid'
    elif 'no finding' in l:
        return 'normal'
    else:
        return 'other'

    
def src1_imgpath(patient):
    _ = os.path.exists(os.path.join(INPUT_PATH_1_IMG, f'{patient}.jpg'))
    return f'{patient}.jpg' if _ else f'{patient}.png'


def src2_label(l):
    return src1_label(l)


def src0_label(l):
    l = l.lower()
    if (('pneumonia' in l) and ('covid' in l)):
        return 'covid'
    elif (('pneumonia' in l) and ('covid' not in l)):
        return 'pneumonia'
    elif ('no finding' in l):
        return 'normal'
    else:
        return 'other'