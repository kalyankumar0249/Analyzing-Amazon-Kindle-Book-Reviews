import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
import os
import re
import string

#sc =SparkContext()
sqlContext = SQLContext(sc)
data_rdd = sc.textFile("D:/Study/Sem2/CS5123/project/sample.txt")
parts_rdd = data_rdd.map(lambda l: l.split("\t"))
data = parts_rdd.map(lambda x: [x[0],int(x[2]),x[3]])
#removing unnecessary features
def remove_features(data_str):
    # compile regex
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    # convert to lowercase
    data_str = data_str.lower()
    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    # remove @mentions
    data_str = mention_re.sub(' ', data_str)
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    # remove numeric 'words'
    data_str = num_re.sub(' ', data_str)
    # remove non a-z 0-9 characters and words shorter than 3 characters
    list_pos = 0
    cleaned_str = ''
    for word in data_str.split():
        if list_pos == 0:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = word
            else:
                cleaned_str = ' '
        else:
            if alpha_num_re.match(word) and len(word) > 2:
                cleaned_str = cleaned_str + ' ' + word
            else:
                cleaned_str += ' '
        list_pos += 1
    return cleaned_str
#remove_features('jasdfks sbjfnkas k2nejnc asjjdn')
data1 = data.map(lambda x: [x[0],x[1],remove_features(x[2])])
#Checking for the blank rows
def check_blanks(data_str):
    is_blank = str(data_str[2].isspace())
    if is_blank == 'False':
        return data_str
data5 = data1.map(check_blanks)#.filter(lambda x: x[2] != 'False').take(5)
#positive and negative encoding of the rating. Rating with 4 or 5 is given positive as 1 and 1,2,3 is given negative as 0
def sentiment(data_str):
    if data_str > 3:
        return 1
    else:
        return 0
data6 = data5.map(lambda x: [x[0],float(sentiment(x[1])),x[2]])
data_df = sqlContext.createDataFrame(data6, ["id", "label", "text"])
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
# stop words
add_stopwords = ["book","story","read","http","https","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=20000, minDF=5)
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])
pipelineFit = pipeline.fit(data_df)
data = pipelineFit.transform(data_df)
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed = 12345)
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

