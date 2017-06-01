# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:59:16 2017

"""
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


def preprocess(df):
    df['tokens'] = df['docs'].apply(preprocess_doc)
    # df['tokens'] = df['tokens'].apply(lowering)
    return df


def sense2vec_style(token):
    text = token.text.replace(' ', '_')
    return '{0}'.format(text)
    # tag = token.ent_type_ or token.pos_
    # return '{0}|{1}'.format(text, tag)


def entity_or_lemma(t):
    text = t.text if t.ent_type else t.lemma_
    return text.replace(' ', '_')


def preprocess_doc(doc):
    # merge entities
    for ent in doc.ents:
        if len(ent) > 1:
            ent.merge(ent.root.tag_, ent.text, ent.label_)

    # merge noun chunks
    # for nc in doc.noun_chunks:
    #    nc.merge(nc.root.tag_, nc.text, nc.root.ent_type_)

    # return our tokens
    return [entity_or_lemma(tok) for tok in doc if not filter_token(tok)]


def filter_token(tok):
    return tok.is_stop or tok.is_punct or tok.pos_ in ["PUNCT", "SYM", "NUM"] \
           or tok.tag_ in ["POS", "VBZ", "CD", "VB", "RB"] \
           or tok.like_email or tok.is_space or tok.like_num or tok.text in ['+'] \
           or tok.lower_ in ENGLISH_STOP_WORDS
