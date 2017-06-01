# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:24:59 2017

"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# flatten topics to one-hot encoding (dummy variables)
def load_labels(df):
    f = 'data/topicDictionary.txt'
    topics_df = pd.read_csv(f, header=None)
    topics_df.columns = ['topics']
    desired_topics = list(topics_df['topics'])

    df['Filtered_Topics'] = df['Topics'].apply(lambda topics: [t for t in topics if t in desired_topics])

    # classes keep all in order of the topic dictionary # remove it when creating sampling_df
    mlb = MultiLabelBinarizer(classes=list(topics_df['topics']))
    Y = mlb.fit_transform(df['Filtered_Topics'])
    print(mlb.classes_)
    return Y

# p = np.asarray(mlb.classes_)
# p = p.astype(str)
