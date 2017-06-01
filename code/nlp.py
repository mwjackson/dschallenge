# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:03:21 2017
"""
import pandas as pd
import spacy
from spacy import attrs


f = 'data/sample_data4_1.csv'
t = 'data/TestData.csv'

nlp = spacy.load('en_core_web_md')
nlp.vocab.add_flag(lambda s: s in spacy.en.word_sets.STOP_WORDS, spacy.attrs.IS_STOP)


def load_data(quick=False, test=False):
    df = pd.read_csv(f, encoding='latin')[:10000] if quick \
        else pd.read_csv(f, encoding='latin')

    print('using {0} samples'.format(len(df)))

    df['BodyText'] = df['BodyText'].fillna('')
    df['Topics'] = df['Topics'].apply(eval)
    df['docs'] = [doc for doc in nlp.pipe(df['BodyText'], batch_size=1000, n_threads=-1)]

    dft = None
    if test:
        dft = pd.read_csv(t, encoding='utf8')
        dft['BodyText'] = dft['BodyText'].fillna('')
        dft['Topics'] = dft['Topics'].apply(eval)
        dft['docs'] = [doc for doc in nlp.pipe(dft['BodyText'], batch_size=1000, n_threads=-1)]

    return df, dft
