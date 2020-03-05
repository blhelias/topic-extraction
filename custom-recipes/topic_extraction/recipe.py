# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import *

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from utils.preprocessing_utils import clean_text, build_topic_mapper

## PARAMETRES
DESCRIPTION_COL = get_recipe_config()['description']
PRODUCT_COL = get_recipe_config()['product']
NB_TOPIC = int(get_recipe_config()['n_topic'])
NB_WORD_BY_TOPIC = int(get_recipe_config()['top_word'])
MAX_DF=float(get_recipe_config()['max_df'])

## INPUT DATASET
df_input_name = get_input_names_for_role('input')[0]
ds = dataiku.Dataset(df_input_name)
df = ds.get_dataframe().iloc[0:100]

## Creation du corpus
df[DESCRIPTION_COL] = df[DESCRIPTION_COL].fillna(' ')
corpus = df[DESCRIPTION_COL]

## TFIDF
tfidf_vectorizer = TfidfVectorizer(max_df=MAX_DF)
tfidf = tfidf_vectorizer.fit_transform(corpus)
description_features = tfidf_vectorizer.inverse_transform(tfidf)
features_list = tfidf_vectorizer.get_feature_names()

df_product_word = pd.DataFrame(
    tfidf.toarray(),
    index=df[PRODUCT_COL],
    columns=features_list
)

## NMF
nmf = NMF(n_components=NB_TOPIC, random_state=1).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
topic_map = build_topic_mapper(nmf, tfidf_feature_names, NB_WORD_BY_TOPIC)


df_topic = pd.DataFrame(np.argmax(nmf.transform(tfidf), axis=1))
df_topic.columns=['topic']

df_topic["topic"] = df_topic.topic.apply(lambda x: topic_map[x])
df_output = pd.concat([df,df_topic], axis=1)

## OUTPUTS
output = get_output_names_for_role('main_output')[0]
df_enriched = dataiku.Dataset(output)
df_enriched.write_with_schema(df_output)
