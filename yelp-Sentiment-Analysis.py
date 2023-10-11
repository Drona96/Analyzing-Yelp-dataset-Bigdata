#!/usr/bin/env python
# coding: utf-8

# In[38]:


import string
import re
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("yelp").getOrCreate()
# remove punctuation
def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    return nopunct

# binarize rating
def convert_rating(rating):
    rating = int(rating)
    if rating >=4: return 1
    else: return 0

from pyspark.sql.functions import udf
from pyspark.sql.functions import col


# In[39]:


schema = "review_id STRING, user_id STRING, business_id STRING, stars INT, date timestamp, text STRING, useful INT, funny INT, cool INT"
df = spark.read.option('multiline',True).option('delimeter',',').schema(schema).csv('gs://yelp-dataset-bucket/yelp_review.csv',header = True)

df.show(4,False)
print(df.count())
df = df.withColumn('stars',col('stars').cast("integer"))
df = df.where(df.stars.isNotNull())
df = df.where(df.text.isNotNull())
df.count()


# In[40]:


from pyspark.sql.types import StringType

punct_remover = udf(lambda x: remove_punct(x))
rating_convert = udf(lambda x: convert_rating(x))
#select 1.5 mn rows of reviews text and corresponding star rating with punc removed and ratings converted

# df = df.withColumn('text',col('text').cast(StringType))
df = df.withColumn('text', punct_remover(col('text')))
df = df.withColumn('stars', rating_convert('stars'))
df.show(10,False)


# In[41]:


from pyspark.ml.feature import *
from pyspark.ml import Pipeline
#tokenizer and stop word remover
tok = Tokenizer(inputCol="text", outputCol="words")
#stop word remover
stopwordrm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
# Build the pipeline 
pipeline = Pipeline(stages=[tok, stopwordrm])
# Fit the pipeline 
review_tokenized = pipeline.fit(df).transform(df).cache()


# In[42]:


df.printSchema()


# In[ ]:


from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import CountVectorizer
# import org.apache.spark.mllib.linalg.Vectors
from pyspark.mllib.classification import SVMWithSGD
# add ngram column
n = 3
ngram = NGram(inputCol = 'words', outputCol = 'ngram', n = n)
add_ngram = ngram.transform(review_tokenized)
# count vectorizer and tfidf
cv_ngram = CountVectorizer(inputCol='ngram', outputCol='tf_ngram')
cvModel_ngram = cv_ngram.fit(add_ngram)
cv_df_ngram = cvModel_ngram.transform(add_ngram)
# create TF-IDF matrix
idf_ngram = IDF().setInputCol('tf_ngram').setOutputCol('tfidf_ngram')
tfidfModel_ngram = idf_ngram.fit(cv_df_ngram)
tfidf_df_ngram = tfidfModel_ngram.transform(cv_df_ngram)
# split into training & testing set
splits_ngram = tfidf_df_ngram.select(['tfidf_ngram', 'label']).randomSplit([0.8,0.2],seed=100)
train_ngram = splits_ngram[0].cache()
test_ngram = splits_ngram[1].cache()
# Convert feature matrix to LabeledPoint vectors
train_lb_ngram = train_ngram.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
test_lb_ngram = train_ngram.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
# fit SVM model of only trigrams
numIterations = 50
regParam = 0.3
svm = SVMWithSGD.train(train_lb_ngram, numIterations, regParam=regParam)
#extract top 20 trigrams based on weights
top_ngram = svm_coeffs_df_ngram.sort_values('weight')['ngram'].values[:20]
bottom_ngram = svm_coeffs_df_ngram.sort_values('weight', ascending=False)['ngram'].values[:20]
ngram_list = list(top_ngram) + list(bottom_ngram)


# In[23]:


# replace the word with selected ngram
def ngram_concat(text):
    text1 = text.lower()
    for ngram in ngram_list:
        if ngram in text1:
            new_ngram = ngram.replace(' ', '_')
            text1 = text1.replace(ngram, new_ngram)
    return text1
ngram_df = udf(lambda x: ngram_concat(x))
ngram_df = review_tokenized.withColumn('text', (ngram_df('text')))


# In[25]:


# count vectorizer and tfidf
from pyspark.mllib.feature import IDF
cv = CountVectorizer(inputCol='words_nsw', outputCol='tf')
cvModel = cv.fit(review_tokenized)
count_vectorized = cvModel.transform(review_tokenized)
tfidfModel = IDF.fit(count_vectorized)
tfidf_df = tfidfModel.transform(count_vectorized)


# In[26]:


# split into training and testing set
splits = tfidf_df.select(['tfidf', 'label']).randomSplit([0.8,0.2],seed=100)
train = splits[0].cache()
test = splits[1].cache()

numIterations = 50
regParam = 0.3
svm = SVMWithSGD.train(train_lb, numIterations, regParam=regParam)
test_lb = test.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
scoreAndLabels_test = test_lb.map(lambda x: (float(svm.predict(x.features)), x.label))
score_label_test = spark.createDataFrame(scoreAndLabels_test, ["prediction", "label"])


# In[27]:


# Elastic Net Logit
lambda_par = 0.02
alpha_par = 0.3
lr = LogisticRegression().        setLabelCol('label').        setFeaturesCol('tfidf').        setRegParam(lambda_par).        setMaxIter(100).        setElasticNetParam(alpha_par)
lrModel = lr.fit(train)
lr_pred = lrModel.transform(test)


# In[ ]:




