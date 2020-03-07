import pandas as pd
import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer
import sys
np.set_printoptions(threshold=np.inf)
f = open('lda_output.txt', 'a')
sys.stdout = f

text = pd.read_csv('top_review.csv', usecols=[1, 2])
text = pd.DataFrame(text)
title = text[['review_headline']]
body = text[['review_body']]
corpus = []
for i in range(4376):
    temp = (str(body.loc[i]))[15:].replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ')
    corpus.append(temp)
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus).toarray()
feature_name = tuple(vectorizer.get_feature_names())
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)
topic_word = model.topic_word_  # 每个topic的词矩阵，每行表示一个话题的词概率，和为1
print(topic_word)
'''
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(feature_name)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
'''
