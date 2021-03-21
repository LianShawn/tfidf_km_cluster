# tfidf_km_cluster

#import packages
```
import jieba
import pandas as pd
import jieba.posseg as pseg
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
```

#import stopwords
```
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords = stopwordslist('stopwords-zh.txt')
```
#import data and clean it
```
data = pd.read_csv('high_school.csv')
data = data.dropna()
data.index = range(len(data))
pattern = re.compile(r'[^\u4e00-\u9fa5]')
data['question'] = data['question'].apply(lambda x: re.sub(pattern,'',x))
```
#collect all of the content into one list
```
texts = []
for i in data['question']:
    texts.append(i)
```
#train the model
```
tfidif_vec = TfidfVectorizer(stop_words = stopwords,tokenizer = jieba.lcut)
X = tfidif_vec.fit_transform(texts)
result = pd.DataFrame(X.toarray(),columns = tfidif_vec.get_feature_names())
```
#the result of the tfidf
```
tfidf_sum = result.T.sum(axis=1)
da = tfidf_sum.sort_values(ascending = False)
da = pd.DataFrame(da)
```
#save it as excel
```
da[0:200].to_excel('high_frequent_word.xlsx')
```

#cluster
```
n_cluster_num = 4
km = KMeans(n_clusters = n_cluster_num)
km.fit(X)

print('Top terms per cluster:')
order_centroids = km.cluster_centers_.argsort()[:,::-1]
terms = tfidif_vec.get_feature_names()
for i in range(n_cluster_num):
    print('X')
    top_ten_words = [terms[ind] for ind in order_centroids[i,:40]]
    print('Cluster {}:{}'.format(i,' '.join(top_ten_words)))
    
```

#count the number of each category
```
data['category'] = km.labels_

num = Counter()
for i in data['category']:
    num[i]+=1
    
num_result = num.most_common(10)
print(num_result)
```
#save it
```
data.to_excel('high_category.xlsx')
```

