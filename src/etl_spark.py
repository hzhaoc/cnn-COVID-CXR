#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
etl-spark.py: do ETL on source images with spark
see ./src/etl.py for data sources
"""

__author__ = "Hua Zhao"

from src.etl import *
from src.utils_spark import CXR
import pyspark


spark = pyspark.sql.SparkSession \
    .builder \
    .appName("COVID-CXR META ETL") \
    .config("spark.some.config.option", "xxx") \
    .getOrCreate()

try:
    # once intialized, it's global in the jupyter notebook kernal; one context per notebook
    sc = pyspark.SparkContext.getOrCreate('local[*]')
except:
    sc = pyspark.SparkContext('local[*]')
# sqlc = pyspark.SQLContext


def spark_etl():
    rdd0 = spark_etl_META_0()
    rdd1 = spark_etl_META_1()
    rdd2 = spark_etl_META_2()
    rdd3 = spark_etl_META_3()
    rdd4 = spark_etl_META_4()
    rdd = sc.union([rdd0, rdd1, rdd2, rdd3, rdd4])
    CXRs = rdd.collect()  # Action, to cache in pandas.dataframe
    META = pd.DataFrame()
    for i, cxr in enumerate(CXRs):
        META.loc[i, 'patient'] = cxr.pid
        META.loc[i, 'img'] = cxr.fn_src
        META.loc[i, 'label'] = cxr.label
        META.loc[i, 'src'] = cxr.src
    META['src'] = META.src.astype(int)
    print('done')
    return META


def spark_etl_META_0():
    # src 0
    rdd = spark.read.csv(INPUT_PATH_0_META, header=True, sep=",").rdd
    rdd = rdd.filter(lambda x: x.view in (["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]))
    
    global _src0_url
    _src0_url = rdd.map(lambda x: x.url).collect()  # RDD ACTION, for duplicates in src 3
    
    rdd = rdd.map(lambda x: CXR(pid=x.patientid, fn_src=x.filename, label=x.finding, src=0))
    rdd = rdd.filter(lambda x: 1 - x.isna())
    rdd = rdd.filter(lambda x: x.label!= 'other')
    return rdd


def spark_etl_META_1():
    # src 1
    rdd = spark.read.csv(INPUT_PATH_1_META, header=True, sep=",").rdd
    rdd = rdd.map(lambda x: CXR(pid=x.patientid, fn_src=None, label=x.finding, src=1))
    rdd = rdd.filter(lambda x: 1- x.isna())
    rdd = rdd.filter(lambda x: x.label!= 'other')
    return rdd


def spark_etl_META_2():
    # src 2
    rdd = spark.read.csv(INPUT_PATH_2_META, header=True, sep=",").rdd
    rdd = rdd.map(lambda x: CXR(pid=x.patientid, fn_src=x.imagename, label=x.finding, src=2))
    rdd = rdd.filter(lambda x: 1- x.isna())
    rdd = rdd.filter(lambda x: x.label!= 'other')
    return rdd


def spark_etl_META_3():
    # src 3
    def _(metapath, label):
        rdd = spark.read.csv(metapath, header=True, sep=",").rdd
        if label == 'covid':
            # https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx.ipynb
            rdd = rdd.filter(lambda x: x['FILE NAME'] not in discard)
            rdd = rdd.filter(lambda x: x.URL not in _src0_url)
        rdd = rdd.map(lambda x: CXR(pid=x['FILE NAME'], fn_src=f"{x['FILE NAME']}.{x.FORMAT.lower()}", label=label, src=3))
        rdd = rdd.filter(lambda x: 1- x.isna())
        return rdd
    rdd0 = _(INPUT_PATH_3_0_META, 'covid')
    rdd1 = _(INPUT_PATH_3_1_META, 'normal')
    rdd2 = _(INPUT_PATH_3_2_META, 'pneumonia')
    return sc.union([rdd0, rdd1, rdd2])


def spark_etl_META_4():
    # src 4
    rdd_1 = spark.read.csv(INPUT_PATH_4_META_1, header=True, sep=",").rdd
    rdd = spark.read.csv(INPUT_PATH_4_META, header=True, sep=",").rdd
    # one img per patient, one patient per row
    rdd_1 = rdd_1.distinct()
    rdd = rdd.distinct()
    rdd_1 = rdd_1.map(lambda x: (x.patientId, x['class']))
    rdd = rdd.map(lambda x: (x.patientId, x.Target))
    rdd = rdd_1.leftOuterJoin(rdd)
    # according to src 4, kaggle file description, non-pneumonia has classes of normal and non-normal
    # for this project to classify [normal, pneumonia, COVID], we drop non-normal-non-pneumonia
    rdd = rdd.filter(lambda x: (x[1][1]=='1')|((x[1][1]=='0')&(x[1][0]=='Normal')))
    rdd = rdd.map(lambda x: CXR(pid=x[0], fn_src=None, label=x[1][1], src=4))
    return rdd
