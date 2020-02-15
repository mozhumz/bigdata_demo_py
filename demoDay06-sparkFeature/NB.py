# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import jieba
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import HashingTF,StringIndexer,Tokenizer,IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 解决编码问题
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 创建sparkSession
spark = SparkSession.builder.appName("NB Test").enableHiveSupport().getOrCreate()

# 读取hive的数据
df = spark.sql("select sentence,label from badou.news_noseg")
df.show()

# 定义结巴切词方法
def seg(text):
    return ' '.join(jieba.cut(text,cut_all=True))
seg_udf = udf(seg, StringType())

# 对数据进行结巴切词
df_seg = df.withColumn('seg',seg_udf(df.sentence)).select('seg','label')
df_seg.show()
# 将分词做成ArrayType（）
tokenizer = Tokenizer(inputCol='seg',outputCol='words')
df_seg_arr=tokenizer.transform(df_seg).select('words','label')
df_seg_arr.show()

# 切词之后的文本特征的处理
tf = HashingTF(numFeatures=1<<18,binary=False,inputCol='words',outputCol='rawfeatures')
df_tf = tf.transform(df_seg_arr).select('rawfeatures','label')
df_tf.show()

idf = IDF(inputCol='rawfeatures',outputCol='features')
idfModel = idf.fit(df_tf)
df_tfidf = idfModel.transform(df_tf)
df_tfidf.show()

# label数据的处理
stringIndexer = StringIndexer(inputCol='label',outputCol="indexed", handleInvalid='error')
indexer = stringIndexer.fit(df_tfidf)
df_tfidf_lab = indexer.transform(df_tfidf).select('features','indexed')
df_tfidf_lab.show()

# 切分训练集和预测集
splits = df_tfidf_lab.randomSplit([0.7, 0.3], 1234)
train = splits[0]
test = splits[1]

# 定义模型
nb = NaiveBayes(featuresCol="features", labelCol="indexed", predictionCol="prediction",
                probabilityCol="probability", rawPredictionCol="rawPrediction",
                smoothing=1.0,
                modelType="multinomial")
# 模型训练
model = nb.fit(train)
# 测试集预测
predictions = model.transform(test)
predictions.show()

# 计算准确率
evaluator = MulticlassClassificationEvaluator(labelCol="indexed",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

