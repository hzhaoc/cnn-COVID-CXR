#!/usr/bin/env python
# https://stackoverflow.com/questions/2429511/why-do-people-write-usr-bin-env-python-on-the-first-line-of-a-python-script

"""
etl-spark.py: do ETL on source images with spark
"""

__author__ = "Hua Zhao"

from src.etl import *
from src.rdd import CXR


def spark_etl_META_0():
    # src 0
    rdd = spark.read.csv(INPUT_PATH_0_META, header=True, sep=",").rdd
    global _src0_url
    _src0_url = rdd.map(lambda x: x.url).collect()  # RDD ACTION, for duplicates in src 3
    rdd = rdd.filter(lambda x: x.view in (["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]))
    rdd = rdd.map(lambda x: CXR(pid=x.patientid, fn_src=x.filename, label=x.finding, src=0))
    rdd = rdd.filter(lambda x: 1 - x.isna())
    rdd = rdd.filter(lambda x: x.label!= 'other')
    return rdd

