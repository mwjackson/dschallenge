# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:30:22 2017

"""

import os
from time import sleep
import pandas as pd
import json
import numpy as np
from itertools import islice

import requests
import io  # extracts data from the web

api_key = os.getenv('nyt_api_key')

# qs = [
#    'australia gun control',
#    'bastille day truck attack',
#    'berlin christmas market attack',
#    'brussels attacks',
#    'charlie hebdo attack',
#    'france train attack',
#    'munich shooting',
#    'orlando terror attack',
#    'paris attacks',
#    'san bernardino shooting',
#    'tunisia attack 2015',
#    'turkey coup attempt',
#    'zika virus',
# ]

qs = [
    'darknet',
    'sydney siege',
    'australian security counter terrorism',
    'london woolwich attack',
    'biometrics',
    'rio+20 earth summit',
    'al-shabaab',
    'cabinet office briefing room',
    'deflation'
]

for q in qs:
    docs = []

    for page in range(0, 10):
        r = requests.get(
            'https://api.nytimes.com/svc/search/v2/articlesearch.json?api-key={0}&q={1}&page={2}'.format(api_key, q,
                                                                                                         page))
        print(r.headers)
        docs += [json.dumps(doc) for doc in r.json().get('response', {}).get('docs', [])]
        sleep(1)  # api is limited to 1 call per second

    with io.open('{0}.json'.format(q), 'w', encoding='utf-8') as f:
        f.write(u'[')
        f.write(u','.join(docs))
        f.write(u']')

import pandas as pd
import json
import glob
import os

directory = "*.json"


def json_to_df(path, topic):
    print('loading topic {0} - {1}'.format(topic, path))
    with open(path, 'r') as f:
        line = json.loads(next(f))

    df = pd.DataFrame(line)
    df['abstract'].fillna('', inplace=True)
    df['lead_paragraph'].fillna('', inplace=True)
    df['Topics'] = '["{0}"]'.format(topic)
    # df['Topics'] = df['Topics'].apply(eval)
    df['BodyText'] = df['abstract'] + ' ' + df['lead_paragraph']
    return df


dfs = []
for path in glob.glob(directory):
    file, ext = os.path.splitext(path)
    dfs.append(json_to_df(path, file.replace(' ', '')))

df = pd.concat(dfs)

df['section_name'] = df['section_name'].fillna('')
df['strTopics'] = df['Topics']
df['Topics'] = df['Topics'].apply(eval)


# df['strTopics'] = df['strTopics'].astype(str)

#####

def check_contains_label(df):
    for index, row in df.iterrows():
        if 'cabinet office briefing room' in row['strTopics']:
            row['Topics'].append('cobra')


check_contains_label(df)

top = df[df['strTopics'].str.contains('woolwichattack')]
top = top[top['BodyText'].str.contains('Woolwich')]

####################################

df = df[df['section_name'] != 'Sports']
df = df.drop_duplicates(subset='BodyText')
df.rename(columns={'pub_date': 'WebPubDate'}, inplace=True)
df['WebPubDate'] = df['WebPubDate'].fillna('')
df['WebPubDate'] = df['WebPubDate'].apply(lambda x: x.split('T')[0])  # split returns a list
df['WebPubDate'] = df['WebPubDate'].apply(lambda x: x.replace('-', '/'))

df = df[["BodyText", "Topics", "WebPubDate"]]

df.to_csv('data/NYtimes2.csv')

df = pd.read_csv('data/NYtimes.csv', encoding='latin')
