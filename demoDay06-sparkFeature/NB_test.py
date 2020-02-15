# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import jieba
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')
spark = SparkSession.builder.appName("NB Test").enableHiveSupport().getOrCreate()

df = spark.sql("select content,label from badou.news_no_seg limit 10")
df.show()
def seg(text):
    return ' '.join(jieba.cut(text,cut_all=True))

seg_udf = udf(seg,StringType())

df1 = df.withColumn('seg',seg_udf(df.content))
df1.show()