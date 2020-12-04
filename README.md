# COVID-CXR-Classification
This project built a pipeline with multiple trained models to classify Chest X-Ray images into Normal/Penumonia/COVID-19. 

## Features
- **DVC pipeline** (with simple CLI for setup and run) to reproduce and version control the whole process
- **Segmentation** and **Adaptive Histogram Equilization** with OpenCV in preprocess
- Over **20,000** CXRs and labels
- **Visualization** of image transformation for clarification
- Trained models including **ResNet**, **VGGNet**, [COVID-Net](https://github.com/lindawangg/COVID-Net) with tensorflow and pytorch.
- **Augmentation** in training
- **Grad-CAM Visualization** of model feature for clarification
- **Hyperparameter** tunning for ETL, training, evaluating, models, visualization.


## Environment
Main software packages include:
```
  - conda=4.9.1
  - python=3.6
  - pyspark=3.0.1
  - pytorch=1.3.1
  - tensorflow=2.3.1
  - dvc==1.10.2
  - torchvision=0.4.2
  - scikit-learn
  - numpy
  - scipy
  - pandas
  - matplotlib=3.1.0
  - pydicom
  - opencv
  - ipython
  - notebook
  - jupyter
  - ipykernel
  - pip
```

## Setup
0. Set up Python, Anaconda, Git, a cloned copy of the project
1. Create environment with
```
conda env create -f env.yml
```
2. For Windows OS, build pipeline with
```
setup
```

## Usage
For Windows OS, run pipeline with
```
run
```

## Pipeline
The pipeline is built on DVC ([Data Version Control](https://dvc.org/doc/start)). It caches and versions data flow, constructs a DAG ([directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)) used to reproduce the whole procedure. The DAG consists of series of ordered stages with dependenceis and outputs including hyperparameter setting.

### DAG
![DAG](DAG.png)

### Hyperparameters
`params.yaml` is the hyperparameter file to construct pipeline in DVC, and for user to fine-tune the whole process from ETL, model setup to training and visualization.
It fine-tunes:
- ETL: image size, crop area, spark control, segmentation control, adaptive or global histogram equilizaiton, etc.
- Model: model tool (tensorflow/pytorch), model name, model architect (VGG, ResNet, COVID-Net), transfer-learning control, etc.
- Train: epoch, learning rate, batch size, COVID-19 label weights in batch, COVID-19 label weights in optimization, etc.
- Visualization: example number, figure size for both transformation and model feature.

### Data
#### Data Source
The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)

### Preprocess
Use `PySpark` to do ETL process with image meta. Transform image data including histogram equilization and segmentaiton with `OpenCV` and prepare image data ready for `PyTorch`.
See `./src/etl.py`, `./src/etl_spark.py` for Spark ETL.
See `./src/transform.py` for image pre-training transform.

#### Data description:
```python
{'pneumonia': 11092, 'normal': 10340, 'covid': 617}
```

#### Pretraining Transform
![Transform Example](https://github.com/hzhaoc/COVID-CXR/blob/main/diagnosis/transform/transform%20example.png)

### Model
Supported models include `VGG11`, `VGG19`, `ResNet18`, `ResNet50`, `COVID-Net-CXR-A`, `COVID-Net-CXR-Large`, `COVID-Net-CXR-Small`.

#### Trained model and weights
For COVID-Net tensorflow models, access metagraph and checkpoints source from https://github.com/lindawangg/COVID-Net. 
For VGGNet, ResNet pytorch models, access saved model from `./model/`

### Train and Evaluate


## Issues

## Todo
